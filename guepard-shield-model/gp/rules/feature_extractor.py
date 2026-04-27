"""Extract eBPF-friendly features from syscall windows."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter


class WindowFeatureExtractor:
    """
    Extract features from windows of syscall token IDs.

    Features (all eBPF-computable):
    1. Syscall frequency histogram (len(vocab) features)
    2. Top discriminative bigrams selected via mutual information
    3. Dangerous-syscall rate (execve, connect, socket, openat, etc.)
    """

    DANGEROUS_SYSCALLS = {
        'execve', 'connect', 'socket', 'openat', 'open',
        'chmod', 'chown', 'kill', 'ptrace', 'setuid',
        'setgid', 'ioctl', 'mmap', 'mprotect', 'dup2',
    }

    def __init__(self, vocab_path: Path | str, top_ngrams: int = 100):
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
        self.top_ngrams = top_ngrams
        self.dangerous_indices = {
            i for i, name in enumerate(self.vocab) if name in self.DANGEROUS_SYSCALLS
        }
        self.ngram_to_idx: Dict[Tuple[int, ...], int] = {}
        self.ngram_list: List[Tuple[int, ...]] = []

    def _load_vocab(self, vocab_path: Path | str) -> List[str]:
        with open(vocab_path) as f:
            return [line.strip() for line in f if line.strip()]

    def fit_ngrams(self, windows: np.ndarray, labels: np.ndarray) -> None:
        """Select top-K discriminative bigrams via mutual information."""
        from sklearn.feature_selection import mutual_info_classif

        bigram_counts: Counter = Counter()
        pos_bigram_counts: Counter = Counter()

        for window, label in zip(windows, labels):
            valid = window[window != 0]
            if len(valid) < 2:
                continue
            for i in range(len(valid) - 1):
                bg = (int(valid[i]), int(valid[i + 1]))
                bigram_counts[bg] += 1
                if label == 1:
                    pos_bigram_counts[bg] += 1

        candidates = [
            bg for bg, count in bigram_counts.items()
            if count >= 5 and pos_bigram_counts[bg] >= 2
        ]

        if not candidates:
            self.ngram_to_idx = {}
            self.ngram_list = []
            return

        bg_idx = {bg: i for i, bg in enumerate(candidates)}
        X_bg = np.zeros((len(windows), len(candidates)), dtype=np.int32)

        for s, window in enumerate(windows):
            valid = window[window != 0]
            if len(valid) < 2:
                continue
            for i in range(len(valid) - 1):
                bg = (int(valid[i]), int(valid[i + 1]))
                if bg in bg_idx:
                    X_bg[s, bg_idx[bg]] += 1

        mi = mutual_info_classif(X_bg, labels, random_state=42)
        top_k = min(self.top_ngrams, len(candidates))
        top_idx = np.argsort(mi)[::-1][:top_k]

        self.ngram_list = [candidates[i] for i in top_idx]
        self.ngram_to_idx = {bg: i for i, bg in enumerate(self.ngram_list)}
        print(f"Selected {len(self.ngram_list)} bigrams out of {len(candidates)} candidates")

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Convert [N, window_size] windows to [N, n_features] feature matrix."""
        n = windows.shape[0]
        n_features = self.vocab_size + len(self.ngram_list) + 1
        X = np.zeros((n, n_features), dtype=np.float32)

        for i in range(n):
            valid = windows[i][windows[i] != 0]
            n_valid = len(valid)
            if n_valid == 0:
                continue

            # IDs may exceed vocab_size (special tokens 99-101); slice to vocab_size
            counts = np.bincount(valid, minlength=self.vocab_size)[:self.vocab_size]
            X[i, :self.vocab_size] = counts

            if self.ngram_list and n_valid >= 2:
                for j in range(n_valid - 1):
                    bg = (int(valid[j]), int(valid[j + 1]))
                    if bg in self.ngram_to_idx:
                        X[i, self.vocab_size + self.ngram_to_idx[bg]] += 1

            dangerous = sum(1 for sid in valid if sid in self.dangerous_indices)
            X[i, -1] = dangerous / n_valid

        return X

    def get_feature_names(self) -> List[str]:
        names = list(self.vocab)
        for bg in self.ngram_list:
            s1 = self.vocab[bg[0]] if bg[0] < self.vocab_size else f"ID{bg[0]}"
            s2 = self.vocab[bg[1]] if bg[1] < self.vocab_size else f"ID{bg[1]}"
            names.append(f"{s1}→{s2}")
        names.append("dangerous_rate")
        return names
