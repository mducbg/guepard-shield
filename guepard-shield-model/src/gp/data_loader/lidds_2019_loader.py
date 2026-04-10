"""LID-DS-2019 dataset loader using lazy streaming.

The LID-DS-2019 dataset contains syscall traces with rich features
(timestamp, uid, pid, tid, process name, syscall, direction, args).
"""

import csv
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from gp.config import LIDDS_2019_DIR
from gp.data_loader.recording import Recording


def _load_csv_metadata(scenario_dir: Path) -> dict[str, dict]:
    """Load per-recording metadata from runs.csv.

    Returns {scenario_name: {"label": int, "recording_time": float, "exploit_start_time": float|None}}
    exploit_start_time is seconds from recording start, or None for normal recordings.
    """
    csv_path = scenario_dir / "runs.csv"
    if not csv_path.exists():
        return {}
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row[" scenario_name"].strip()
            is_exploit = row[" is_executing_exploit"].strip() == "True"
            recording_time = float(row[" recording_time"].strip())
            exploit_start = float(row[" exploit_start_time"].strip())
            result[name] = {
                "label": int(is_exploit),
                "recording_time": recording_time,
                "exploit_start_time": exploit_start if exploit_start >= 0 else None,
            }
    return result


def _parse_lidds_2019_file(filepath: Path) -> tuple[list[str], list[float], list[int]]:
    """Parse a LID-DS-2019 syscall trace file.

    File format (space-separated):
        line_no  HH:MM:SS.ns  cpu  uid  process_name  tid  direction  syscall  [args...]

    Returns:
        Tuple of (syscalls, timestamps, tids) — only syscall-exit events (direction="<").
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
            if len(parts) < 8:
                continue

            try:
                # parts[1] = "HH:MM:SS.nanoseconds"
                time_str = parts[1]
                h, m, rest = time_str.split(":")
                s, ns = rest.split(".")
                timestamp = int(h) * 3600 + int(m) * 60 + int(s) + int(ns) / 1e9

                tid = int(parts[5])
                direction = parts[6]
                syscall = parts[7]

                if direction == "<":
                    syscalls.append(syscall)
                    timestamps.append(timestamp)
                    tids.append(tid)

            except (ValueError, IndexError):
                continue
    return syscalls, timestamps, tids


class LIDDS2019Dataset:
    """Streaming dataset for LID-DS-2019."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize LID-DS-2019 dataset.

        Args:
            data_root: Optional override for dataset directory
            shuffle: Whether to shuffle the recordings
            seed: Random seed for shuffling
        """
        self.data_root = data_root or LIDDS_2019_DIR
        self.shuffle = shuffle
        self.seed = seed
        self._files = self._collect_files()

    def _collect_files(self) -> list[tuple[Path, int, str, dict]]:
        """Collect all file paths with their labels and scenarios."""
        files = []

        if not self.data_root.exists():
            return files

        for scenario_dir in self.data_root.iterdir():
            if not scenario_dir.is_dir():
                continue

            scenario_name = scenario_dir.name
            csv_meta = _load_csv_metadata(scenario_dir)

            for txt_file in scenario_dir.glob("*.txt"):
                meta = csv_meta.get(txt_file.stem, {})
                label = meta.get("label", 0)
                files.append((txt_file, label, scenario_name, meta))

        if self.shuffle and self.seed is not None:
            import random

            rng = random.Random(self.seed)
            rng.shuffle(files)

        return files

    def __iter__(self) -> Iterator[Recording]:
        """Iterate through all recordings (streaming, lazy)."""
        for filepath, label, scenario, csv_meta in self._files:
            syscalls, timestamps, tids = _parse_lidds_2019_file(filepath)
            if syscalls:
                yield Recording(
                    path=str(filepath.relative_to(self.data_root)),
                    label=label,
                    syscalls=syscalls,
                    timestamps=timestamps,
                    tid=tids,
                    metadata={
                        "scenario": scenario,
                        "recording_time": csv_meta.get("recording_time"),
                        "exploit_start_time": csv_meta.get("exploit_start_time"),
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


class LIDDS2019SequenceDataset:
    """Fixed-length sequence dataset for LID-DS-2019."""

    def __init__(
        self,
        sequence_length: int = 100,
        stride: int = 50,
        data_root: Optional[Path] = None,
        vocab_map: Optional[dict[str, int]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize sequence dataset."""
        self.sequence_length = sequence_length
        self.stride = stride
        self.data_root = data_root or LIDDS_2019_DIR
        self.vocab_map = vocab_map or {}
        self.shuffle = shuffle
        self.seed = seed
        self._windows = self._build_window_index()

    def _build_window_index(self) -> list[tuple[Path, int, int, int]]:
        """Build index of all windows from all recordings."""
        files = []
        if not self.data_root.exists():
            return files

        for scenario_dir in self.data_root.iterdir():
            if not scenario_dir.is_dir():
                continue
            csv_meta = _load_csv_metadata(scenario_dir)
            for txt_file in scenario_dir.glob("*.txt"):
                label = csv_meta.get(txt_file.stem, {}).get("label", 0)
                files.append((txt_file, label))

        window_index = []
        for filepath, label in files:
            syscalls, _, _ = _parse_lidds_2019_file(filepath)
            if len(syscalls) >= self.sequence_length:
                for start in range(
                    0, len(syscalls) - self.sequence_length + 1, self.stride
                ):
                    end = start + self.sequence_length
                    window_index.append((filepath, label, start, end))

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

        for filepath, label, start, end in windows:
            syscalls, _, _ = _parse_lidds_2019_file(filepath)
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
