import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import lightning.pytorch as pl
from gp.config import npy_dir

class SyscallDataset(Dataset):
    def __init__(self, split: str, max_windows_per_file: Optional[int] = None):
        self.split = split
        self.max_windows_per_file = max_windows_per_file
        self.split_dir = npy_dir / split
        self.files = sorted(list(self.split_dir.glob("*.npy")))
        
        # We need to know the number of windows in each file for resampling
        self.file_info: List[Tuple[Path, int]] = []
        for f in self.files:
            data = np.load(f, mmap_mode='r')
            self.file_info.append((f, data.shape[0]))
            
        self.index_map: List[Tuple[Path, int]] = []
        self.resample()

    def resample(self):
        """Randomly pick new windows from each file for each epoch."""
        self.index_map = []
        for f, num_windows in self.file_info:
            if self.split == "train" and self.max_windows_per_file:
                if num_windows > self.max_windows_per_file:
                    indices = np.random.choice(num_windows, self.max_windows_per_file, replace=False)
                    for i in indices:
                        self.index_map.append((f, i))
                else:
                    for i in range(num_windows):
                        self.index_map.append((f, i))
            else:
                for i in range(num_windows):
                    self.index_map.append((f, i))
        
        # Shuffle the index map so the model doesn't see windows in file order
        random.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        file_path, window_idx = self.index_map[index]
        data = np.load(file_path, mmap_mode='r')
        window = data[window_idx].astype(np.int64)
        return torch.from_numpy(window)

class SyscallDataModule(pl.LightningDataModule):
    train_ds: SyscallDataset
    val_ds: SyscallDataset
    test_ds: SyscallDataset

    def __init__(self, batch_size: int = 64, max_windows_train: Optional[int] = None):
        super().__init__()
        self.batch_size = batch_size
        self.max_windows_train = max_windows_train

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = SyscallDataset("train", max_windows_per_file=self.max_windows_train)
            self.val_ds = SyscallDataset("val")
        if stage == "test":
            self.test_ds = SyscallDataset("test")

    def on_train_epoch_start(self):
        """Hook called by Lightning at the start of each training epoch."""
        self.train_ds.resample()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
