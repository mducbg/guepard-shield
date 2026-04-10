"""DongTing dataset loader using lazy streaming.

This module provides memory-efficient data loading without loading all data
into memory at once. Uses simple Python generators for streaming.
"""

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from gp.config import DONGTING_DIR
from gp.data_loader.recording import Recording


def _parse_dongting_file(filepath: Path) -> list[str]:
    """Parse a DongTing log file into a list of syscalls."""
    content = filepath.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [s.strip() for s in content.split("|") if s.strip()]


class DongTingDataset:
    """Streaming dataset for DongTing.

    This dataset loads recordings lazily to avoid memory issues with large datasets.

    Example:
        >>> dataset = DongTingDataset()
        >>> for recording in dataset:
        ...     print(f"{recording.path}: {len(recording)} syscalls")
        ...
        >>> # Or use with batching for model training
        >>> batches = dataset.batch(batch_size=32)
        >>> for batch in batches:
        ...     # batch is a list of Recordings
        ...     pass
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize DongTing dataset.

        Args:
            data_root: Optional override for dataset directory
            shuffle: Whether to shuffle the recordings
            seed: Random seed for shuffling
        """
        self.data_root = data_root or DONGTING_DIR
        self.shuffle = shuffle
        self.seed = seed
        self._files = self._collect_files()

    def _collect_files(self) -> list[tuple[Path, int]]:
        """Collect all file paths with their labels (lightweight)."""
        files = []

        normal_dir = self.data_root / "Normal_data"
        if normal_dir.exists():
            for kernel_dir in normal_dir.iterdir():
                if kernel_dir.is_dir():
                    for log_file in kernel_dir.glob("*.log"):
                        files.append((log_file, 0))

        abnormal_dir = self.data_root / "Abnormal_data"
        if abnormal_dir.exists():
            for kernel_dir in abnormal_dir.iterdir():
                if kernel_dir.is_dir():
                    for log_file in kernel_dir.glob("*.log"):
                        files.append((log_file, 1))

        if self.shuffle and self.seed is not None:
            import random

            rng = random.Random(self.seed)
            rng.shuffle(files)

        return files

    def __iter__(self) -> Iterator[Recording]:
        """Iterate through all recordings (streaming, lazy)."""
        for filepath, label in self._files:
            syscalls = _parse_dongting_file(filepath)
            if syscalls:  # Skip empty recordings
                yield Recording(
                    path=str(filepath.relative_to(self.data_root)),
                    label=label,
                    syscalls=syscalls,
                )

    def __len__(self) -> int:
        """Return total number of recordings."""
        return len(self._files)

    def batch(self, batch_size: int) -> Iterator[list[Recording]]:
        """Batch the dataset.

        Args:
            batch_size: Number of recordings per batch

        Yields:
            Lists of Recording objects
        """
        batch = []
        for recording in self:
            batch.append(recording)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class DongTingSequenceDataset:
    """Fixed-length sequence dataset for DongTing.

    Yields fixed-length windows from recordings for sequence modeling.
    """

    def __init__(
        self,
        sequence_length: int = 100,
        stride: int = 50,
        data_root: Optional[Path] = None,
        vocab_map: Optional[dict[str, int]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize sequence dataset.

        Args:
            sequence_length: Length of each sequence window
            stride: Step size between consecutive windows
            data_root: Optional override for dataset directory
            vocab_map: Mapping from syscall names to integer indices
            shuffle: Whether to shuffle the windows
            seed: Random seed for shuffling
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.data_root = data_root or DONGTING_DIR
        self.vocab_map = vocab_map or {}
        self.shuffle = shuffle
        self.seed = seed

        # Build window index (lightweight - just metadata)
        self._windows = self._build_window_index()

    def _build_window_index(self) -> list[tuple[Path, int, int, int]]:
        """Build index of all windows from all recordings."""
        dataset = DongTingDataset(self.data_root)
        windows = []

        for filepath, label in dataset._files:
            syscalls = _parse_dongting_file(filepath)
            if len(syscalls) >= self.sequence_length:
                for start in range(
                    0, len(syscalls) - self.sequence_length + 1, self.stride
                ):
                    end = start + self.sequence_length
                    windows.append((filepath, label, start, end))

        return windows

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
            syscalls = _parse_dongting_file(filepath)
            if len(syscalls) >= end:
                window = syscalls[start:end]
                yield np.array(self._syscalls_to_indices(window), dtype=np.int32), label

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self._windows)

    def batch(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Batch the dataset.

        Args:
            batch_size: Number of windows per batch

        Yields:
            Tuples of (sequences array of shape (batch, seq_len), labels array)
        """
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
