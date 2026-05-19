"""LightningDataModule for Phase 2 syscall anomaly detection."""

from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from gp.dataset import SyscallDataset


class SyscallDataModule(LightningDataModule):
    """Load preprocessed .npy arrays from data/processed/p2/."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        d = self.data_dir
        if stage in ("fit", None):
            self.train_ds = SyscallDataset.from_npy(d / "train_X.npy")
            self.val_ds = SyscallDataset.from_npy(d / "val_X.npy")
        if stage in ("test", None):
            self.test_ds = SyscallDataset.from_npy(
                d / "test_X.npy", d / "test_y.npy"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
