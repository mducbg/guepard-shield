from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from ..config import WindowConfig
from .corpus import DongTingCorpus
from .teacher_dataset import TeacherDataset
from .vocab import SyscallVocab


class TeacherDataModule(L.LightningDataModule):
    """
    LightningDataModule wrapping TeacherDataset for train and validation splits.

    Sequence-level shuffling is preserved: datasets rebuild their flat_index
    each epoch via the DatasetReshuffleCallback (defined in pilot.py / callers).
    """

    def __init__(
        self,
        corpus: DongTingCorpus,
        vocab: SyscallVocab,
        window_config: WindowConfig,
        train_split: str,
        val_split: str,
        batch_size: int = 1024,
        max_windows_per_seq: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.window_config = window_config
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_windows_per_seq = max_windows_per_seq
        self.seed = seed

        self.train_dataset: TeacherDataset = TeacherDataset(
            corpus=corpus,
            vocab=vocab,
            window_config=window_config,
            split_name=train_split,
            batch_size=batch_size,
            shuffle=True,
            max_windows_per_seq=max_windows_per_seq,
            seed=self.seed,
        )
        self.val_dataset: TeacherDataset = TeacherDataset(
            corpus=corpus,
            vocab=vocab,
            window_config=window_config,
            split_name=val_split,
            batch_size=batch_size,
            shuffle=False,
            max_windows_per_seq=max_windows_per_seq,
            seed=self.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # sequence-level shuffle handled by TeacherDataset.reshuffle()
            pin_memory=True,
            # persistent_workers must be False so each epoch forks fresh workers
            # that inherit the reshuffled flat_index from the main process.
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
