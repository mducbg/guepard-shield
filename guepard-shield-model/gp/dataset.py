"""SyscallDataset — torch.Dataset wrapping preprocessed .npy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyscallDataset(Dataset):
    """Wraps token-ID windows stored as int32 .npy arrays.

    For train/val the label array may be absent (unsupervised training).
    For test the label array is required for evaluation.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = X  # kept as numpy (may be mmap)
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx].astype(np.int64))
        if self.y is not None:
            return x, torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x

    @classmethod
    def from_npy(cls, x_path: Path, y_path: Path | None = None) -> SyscallDataset:
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r") if y_path is not None else None
        return cls(X, y)
