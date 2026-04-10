"""LID-DS-2021 dataset loader using lazy streaming.

The LID-DS-2021 dataset contains syscall traces with rich features
and pre-defined train/val/test splits.
"""

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from gp.config import LIDDS_2021_DIR
from gp.data_loader.recording import Recording


def _parse_lidds_2021_sc_file(
    filepath: Path,
) -> tuple[list[str], list[float], list[int]]:
    """Parse a LID-DS-2021 .sc syscall trace file.

    Returns:
        Tuple of (syscalls, timestamps, tids)
    """
    syscalls: list[str] = []
    timestamps: list[float] = []
    tids: list[int] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                timestamp_ns = int(parts[0])
                timestamp = timestamp_ns / 1e9
                tid = int(parts[4])
                syscall = parts[5]
                direction = parts[6]

                if direction == "<":
                    syscalls.append(syscall)
                    timestamps.append(timestamp)
                    tids.append(tid)

            except (ValueError, IndexError):
                continue

    return syscalls, timestamps, tids


def _parse_metadata(json_path: Path) -> dict:
    """Parse LID-DS-2021 metadata JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


class LIDDS2021Dataset:
    """Streaming dataset for LID-DS-2021."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        splits: Optional[list[str]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize LID-DS-2021 dataset.

        Args:
            data_root: Optional override for dataset directory
            splits: List of splits to include (train/val/test). Defaults to all.
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        self.data_root = data_root or LIDDS_2021_DIR
        self.splits = splits
        self.shuffle = shuffle
        self.seed = seed
        self._files = self._collect_files()

    def _collect_files(
        self,
    ) -> list[tuple[Path, Path, int, str, str, str]]:
        """Collect all file paths with metadata."""
        files = []
        splits_to_include = self.splits or ["train", "val", "test"]

        if not self.data_root.exists():
            return files

        for scenario_dir in self.data_root.iterdir():
            if not scenario_dir.is_dir():
                continue

            scenario_name = scenario_dir.name

            for split in splits_to_include:
                split_dir = scenario_dir / split
                if not split_dir.exists():
                    continue

                for entry in split_dir.iterdir():
                    if not entry.is_dir():
                        continue

                    # train/val: recording dirs sit directly under split_dir
                    # test: has intermediate subdirs (normal, normal_and_attack)
                    if (entry / f"{entry.name}.sc").exists():
                        recording_dirs = [entry]
                    else:
                        recording_dirs = [d for d in entry.iterdir() if d.is_dir()]

                    for recording_dir in recording_dirs:
                        recording_name = recording_dir.name
                        sc_file = recording_dir / f"{recording_name}.sc"
                        json_file = recording_dir / f"{recording_name}.json"

                        if not sc_file.exists():
                            continue

                        metadata = _parse_metadata(json_file)
                        label = int(metadata.get("exploit", False))

                        files.append(
                            (
                                sc_file,
                                json_file,
                                label,
                                scenario_name,
                                split,
                                recording_name,
                            )
                        )

        if self.shuffle and self.seed is not None:
            import random

            rng = random.Random(self.seed)
            rng.shuffle(files)

        return files

    def __iter__(self) -> Iterator[Recording]:
        """Iterate through all recordings (streaming, lazy)."""
        for sc_file, json_file, label, scenario, split, recording in self._files:
            syscalls, timestamps, tids = _parse_lidds_2021_sc_file(sc_file)
            if syscalls:
                metadata = _parse_metadata(json_file)
                yield Recording(
                    path=str(sc_file.relative_to(self.data_root).parent),
                    label=label,
                    syscalls=syscalls,
                    timestamps=timestamps,
                    tid=tids,
                    metadata={
                        "scenario": scenario,
                        "split": split,
                        "recording": recording,
                        **metadata,
                    },
                )

    def __len__(self) -> int:
        """Return total number of recordings."""
        return len(self._files)

    def batch(self, batch_size: int) -> Iterator[list[Recording]]:
        """Batch the dataset."""
        batch = []
        for recording in self:
            batch.append(recording)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class LIDDS2021SequenceDataset:
    """Fixed-length sequence dataset for LID-DS-2021."""

    def __init__(
        self,
        sequence_length: int = 100,
        stride: int = 50,
        data_root: Optional[Path] = None,
        splits: Optional[list[str]] = None,
        vocab_map: Optional[dict[str, int]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize sequence dataset."""
        self.sequence_length = sequence_length
        self.stride = stride
        self.data_root = data_root or LIDDS_2021_DIR
        self.splits = splits
        self.vocab_map = vocab_map or {}
        self.shuffle = shuffle
        self.seed = seed
        self._windows = self._build_window_index()

    def _build_window_index(self) -> list[tuple[Path, int, int, int]]:
        """Build index of all windows from all recordings."""
        dataset = LIDDS2021Dataset(self.data_root, self.splits)
        window_index = []

        for sc_file, _, label, _, _, _ in dataset._files:
            syscalls, _, _ = _parse_lidds_2021_sc_file(sc_file)
            if len(syscalls) >= self.sequence_length:
                for start in range(
                    0, len(syscalls) - self.sequence_length + 1, self.stride
                ):
                    end = start + self.sequence_length
                    window_index.append((sc_file, label, start, end))

        return window_index

    def _syscalls_to_indices(self, syscalls: list[str]) -> list[int]:
        """Convert syscall names to integer indices."""
        return [self.vocab_map.get(s, 0) for s in syscalls]

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        """Iterate through all sequence windows."""
        windows = self._windows
        if self.shuffle and self.seed is not None:
            import random

            rng = random.Random(self.seed)
            windows = windows.copy()
            rng.shuffle(windows)

        for sc_file, label, start, end in windows:
            syscalls, _, _ = _parse_lidds_2021_sc_file(sc_file)
            if len(syscalls) >= end:
                window = syscalls[start:end]
                yield np.array(self._syscalls_to_indices(window), dtype=np.int32), label

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self._windows)

    def batch(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Batch the dataset."""
        sequences = []
        labels = []

        for seq, label in self:
            sequences.append(seq)
            labels.append(label)

            if len(sequences) == batch_size:
                yield (
                    np.array(sequences, dtype=np.int32),
                    np.array(labels, dtype=np.int32),
                )
                sequences = []
                labels = []

        if sequences:
            yield np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32)
