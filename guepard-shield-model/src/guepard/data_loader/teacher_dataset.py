import math
import random
from functools import lru_cache
from typing import List, Optional

import keras
import numpy as np

from ..config import WindowConfig
from .corpus import DongTingCorpus
from .vocab import SyscallVocab
from .windowing import get_window_meta, num_sliding_windows


@lru_cache(maxsize=8)
def _read_tokens_from_file(file_path_str: str) -> List[str]:
    """Module-level LRU cache shared across all dataset instances.

    maxsize=8: sequence-level shuffle guarantees all windows of a sequence are
    accessed consecutively, so only O(workers) files need to be hot at once.
    """
    with open(file_path_str, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content.split("|") if content else []


class TeacherDataset(keras.utils.PyDataset):
    """
    Keras PyDataset for the Teacher model.

    Uses a flat index of (seq_idx, win_idx) tuples instead of cumulative-sum
    binary search, which makes uniform sampling per sequence straightforward.

    max_windows_per_seq: if set, randomly sample this many windows from each
    sequence that has more, using uniform sampling over the full sequence length.
    This is the primary mechanism to reduce window-level class imbalance.
    """

    def __init__(
        self,
        corpus: DongTingCorpus,
        vocab: SyscallVocab,
        window_config: WindowConfig,
        split_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        max_windows_per_seq: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.corpus = corpus
        self.vocab = vocab
        self.window_config = window_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_windows_per_seq = max_windows_per_seq
        self._seed = seed

        self.sequences = list(corpus.get_split(split_name))
        self.flat_index: list[tuple[int, int]] = []  # (seq_idx, win_idx)
        self._build_index(random.Random(seed))

    def _build_index(self, rng: random.Random) -> None:
        # Build per-sequence window groups first
        groups: list[list[tuple[int, int]]] = []
        for seq_idx, meta in enumerate(self.sequences):
            n = num_sliding_windows(meta.seq_length, self.window_config)
            if n == 0:
                continue
            win_indices = list(range(n))
            if self.max_windows_per_seq and n > self.max_windows_per_seq:
                win_indices = rng.sample(win_indices, self.max_windows_per_seq)
            groups.append([(seq_idx, w) for w in win_indices])

        # Shuffle at sequence level so all windows of a sequence stay consecutive.
        # This guarantees each file is read once then reused for all its windows —
        # critical for long attack sequences that would thrash a window-level shuffle.
        if self.shuffle:
            rng.shuffle(groups)

        self.flat_index = [pair for group in groups for pair in group]

    def __len__(self) -> int:
        return math.ceil(len(self.flat_index) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._build_index(random.Random())

    def _read_file_tokens(self, file_path_str: str) -> List[str]:
        return _read_tokens_from_file(file_path_str)  # module-level LRU cache

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.flat_index))
        batch_pairs = self.flat_index[start:end]

        pad_id = self.vocab.token2id.get(self.vocab.PAD_TOKEN, 0)
        unk_id = self.vocab.token2id.get(self.vocab.UNK_TOKEN, 0)
        ws = self.window_config.window_size

        batch_x, batch_y = [], []
        for seq_idx, win_idx in batch_pairs:
            seq_meta = self.sequences[seq_idx]
            meta = get_window_meta(
                seq_id=seq_meta.seq_id,
                label=seq_meta.label,
                seq_length=seq_meta.seq_length,
                config=self.window_config,
                file_path=seq_meta.file_path,
                window_idx=win_idx,
            )
            raw_tokens = self._read_file_tokens(str(meta.file_path))
            window_tokens = raw_tokens[
                meta.start_idx : meta.start_idx + meta.window_length
            ]
            ids = [self.vocab.token2id.get(t, unk_id) for t in window_tokens]
            if len(ids) < ws:
                ids.extend([pad_id] * (ws - len(ids)))
            batch_x.append(ids)
            batch_y.append(meta.label)

        return np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32)
