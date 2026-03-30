from pathlib import Path
from typing import Literal, Optional

import keras
import numpy as np

from ..config import WindowConfig
from ..features.vectorizer import SyscallVectorizer
from .corpus import DongTingCorpus
from .teacher_dataset import TeacherDataset
from .windowing import get_window_meta


class SurrogateDataset(keras.utils.PyDataset):
    """
    PyDataset that produces vectorized TF-IDF / N-gram features of the sliding windows
    for the surrogate models exactly following the SurrogateDataset specification.
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vectorizer = vectorizer
        self.label_source = label_source
        self.soft_labels_dir = soft_labels_dir

        self.base_loader = TeacherDataset(
            corpus=corpus,
            vocab=None,  # type: ignore (vocab not used — we use vectorizer)
            window_config=window_config,
            split_name=split_name,
            batch_size=batch_size,
            shuffle=shuffle,
            max_windows_per_seq=max_windows_per_seq,
            seed=seed,
        )

    def __len__(self):
        return len(self.base_loader)

    def on_epoch_end(self):
        self.base_loader.on_epoch_end()

    def _load_label(self, meta, index_in_dataset: int) -> np.ndarray | int:
        if self.label_source == "hard":
            return meta.label
        else:
            if not self.soft_labels_dir:
                raise ValueError("soft_labels_dir must be provided for soft labels")
            # Fast mapping to specific sequence and window from pre-saved npy arrays.
            # Example representation of reading from `.npy` files for soft labels.
            # Depending on how Teacher predictions were saved, this might be a
            # large memmapped array or isolated `.npy` per sequence.
            try:
                # If we saved one giant generic predictions array for the whole split
                # np.load(..., mmap_mode='r')[index_in_dataset]
                # Assuming generic soft_labels.npy exists in directory
                path = self.soft_labels_dir / f"{meta.seq_id}.npy"
                if not path.exists():
                    return meta.label  # fallback
                seq_logits = np.load(path, mmap_mode="r")
                return seq_logits[
                    meta.start_idx // self.base_loader.window_config.stride
                ]
            except Exception:
                return meta.label

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        bs = self.base_loader.batch_size
        start = index * bs
        end = min(start + bs, len(self.base_loader.flat_index))
        batch_pairs = self.base_loader.flat_index[start:end]

        batch_tokens = []
        batch_y = []

        for seq_idx, win_idx in batch_pairs:
            seq_meta = self.base_loader.sequences[seq_idx]
            meta = get_window_meta(
                seq_id=seq_meta.seq_id,
                label=seq_meta.label,
                seq_length=seq_meta.seq_length,
                config=self.base_loader.window_config,
                file_path=seq_meta.file_path,
                window_idx=win_idx,
            )
            raw_tokens = self.base_loader._read_file_tokens(str(meta.file_path))
            window_tokens = raw_tokens[
                meta.start_idx : meta.start_idx + meta.window_length
            ]
            batch_tokens.append(window_tokens)
            batch_y.append(self._load_label(meta, index_in_dataset=seq_idx))

        features = self.vectorizer.transform(batch_tokens)
        return features, np.array(batch_y)
