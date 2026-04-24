"""Stats accumulator for DongTing EDA."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from gp.data_loader.dongting import DongTingRecording

_SPLITS = ("train", "val", "test")
_LABELS = ("normal", "abnormal")


class DongTingStats:
    def __init__(self) -> None:
        # per (split, label) → list of sequence lengths
        self._seq_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)

        # syscall frequency per split
        self._vocab_freq: dict[str, Counter[str]] = defaultdict(Counter)

        # {split: {label: count}}
        self._recording_counts: dict[str, dict[str, int]] = {
            s: {lbl: 0 for lbl in _LABELS} for s in _SPLITS
        }

        # abnormal only: kernel version → count
        self._kernel_version_counts: Counter[str] = Counter()

        # source subdirectory → count
        self._source_counts: Counter[str] = Counter()

    def analyze(self, rec: DongTingRecording) -> None:
        self._recording_counts[rec.split][rec.label] += 1
        self._seq_lengths[(rec.split, rec.label)].append(len(rec.syscalls))
        self._source_counts[rec.source] += 1

        for sc in rec.syscalls:
            self._vocab_freq[rec.split][sc] += 1

        if rec.label == "abnormal":
            self._kernel_version_counts[rec.kernel_version] += 1

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> dict[str, int]:
        return {split: len(counter) for split, counter in self._vocab_freq.items()}

    @property
    def oov_syscalls(self) -> set[str]:
        train_vocab = set(self._vocab_freq.get("train", {}))
        test_vocab = set(self._vocab_freq.get("test", {}))
        return test_vocab - train_vocab

    def seq_length_percentiles(
        self,
        split: str,
        label: str,
        quantiles: list[float],
    ) -> dict[float, int]:
        lengths = self._seq_lengths.get((split, label), [])
        if not lengths:
            return {q: 0 for q in quantiles}
        arr = np.array(lengths)
        return {q: int(np.quantile(arr, q)) for q in quantiles}
