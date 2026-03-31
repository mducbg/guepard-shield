from pathlib import Path
from typing import Literal, Optional

import numpy as np
from torch.utils.data import Dataset

from ..config import WindowConfig
from ..features.vectorizer import SyscallVectorizer
from .corpus import DongTingCorpus
from .teacher_dataset import TeacherDataset, _read_tokens_from_file
from .windowing import extract_window_tokens, get_window_meta


class SurrogateDataset(Dataset):
    """
    torch Dataset that produces vectorized TF-IDF / N-gram features of sliding windows
    for the surrogate models.
    """

    def __init__(
        self,
        corpus: DongTingCorpus,
        vectorizer: SyscallVectorizer,
        window_config: WindowConfig,
        split_name: str,
        label_source: Literal["hard", "soft"] = "hard",
        soft_labels_dir: Optional[Path] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        max_windows_per_seq: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.vectorizer = vectorizer
        self.label_source = label_source
        self.soft_labels_dir = soft_labels_dir

        self.base_loader = TeacherDataset(
            corpus=corpus,
            vocab=None,  # not used — vectorizer handles features
            window_config=window_config,
            split_name=split_name,
            batch_size=batch_size,
            shuffle=shuffle,
            max_windows_per_seq=max_windows_per_seq,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.base_loader.flat_index)

    def reshuffle(self) -> None:
        self.base_loader.reshuffle()

    def _load_label(self, meta) -> np.ndarray | int:
        if self.label_source == "hard":
            return meta.label
        if not self.soft_labels_dir:
            raise ValueError("soft_labels_dir must be provided for soft labels")
        try:
            path = self.soft_labels_dir / f"{meta.seq_id}.npy"
            if not path.exists():
                return meta.label
            return np.load(path, mmap_mode="r")[
                meta.start_idx // self.base_loader.window_config.stride
            ]
        except Exception:
            return meta.label

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        seq_idx, win_idx = self.base_loader.flat_index[index]
        seq_meta = self.base_loader.sequences[seq_idx]

        meta = get_window_meta(
            seq_id=seq_meta.seq_id,
            label=seq_meta.label,
            seq_length=seq_meta.seq_length,
            config=self.base_loader.window_config,
            file_path=seq_meta.file_path,
            window_idx=win_idx,
        )
        raw_tokens = _read_tokens_from_file(str(meta.file_path))
        window_tokens = extract_window_tokens(raw_tokens, meta)

        features = self.vectorizer.transform([window_tokens])
        return features[0], np.array(self._load_label(meta))
