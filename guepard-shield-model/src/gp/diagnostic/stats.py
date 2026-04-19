"""Stats accumulator for LID-DS recording EDA."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from gp.data_loader.recording import Recording

_SPLITS = ("train", "val", "test")


class Stats:
    def __init__(self) -> None:
        self._seq_lengths: dict[str, list[int]] = defaultdict(list)
        self._thread_counts: dict[str, list[int]] = defaultdict(list)
        self._per_thread_lengths: dict[str, list[int]] = defaultdict(list)
        self._vocab_freq: dict[str, Counter[str]] = defaultdict(Counter)
        self._recording_counts: dict[str, dict[str, int]] = {
            s: {"normal": 0, "exploit": 0} for s in _SPLITS
        }
        self._attack_offsets: list[float] = []
        self._attack_fractions: list[float] = []

    def analyze(self, recording: Recording) -> None:
        split = recording.split
        syscalls = recording.syscalls

        # recording counts
        key = "exploit" if recording.is_exploit else "normal"
        self._recording_counts[split][key] += 1

        # sequence length (total exit syscalls)
        self._seq_lengths[split].append(len(syscalls))

        # thread structure
        per_thread: dict[int, int] = defaultdict(int)
        for sc in syscalls:
            per_thread[sc.thread_id] += 1
        self._thread_counts[split].append(len(per_thread))
        for count in per_thread.values():
            self._per_thread_lengths[split].append(count)

        # vocabulary
        for sc in syscalls:
            self._vocab_freq[split][sc.syscall] += 1

        # attack timing — test exploit recordings only
        if recording.is_exploit and recording.exploit_times:
            if len(syscalls) >= 2:
                duration_sec = (
                    (syscalls[-1].timestamp - syscalls[0].timestamp) / 1e9
                )
            else:
                duration_sec = 0.0

            for exploit_time in recording.exploit_times:
                offset = exploit_time - recording.warmup_end
                self._attack_offsets.append(offset)
                if duration_sec > 0:
                    self._attack_fractions.append(offset / duration_sec)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def seq_lengths(self) -> dict[str, list[int]]:
        return dict(self._seq_lengths)

    @property
    def vocab_size(self) -> dict[str, int]:
        return {split: len(counter) for split, counter in self._vocab_freq.items()}

    @property
    def oov_syscalls(self) -> set[str]:
        train_vocab = set(self._vocab_freq.get("train", {}))
        test_vocab = set(self._vocab_freq.get("test", {}))
        return test_vocab - train_vocab

    def seq_length_percentiles(
        self, split: str, quantiles: list[float]
    ) -> dict[float, int]:
        lengths = self._seq_lengths[split]
        if not lengths:
            return {q: 0 for q in quantiles}
        arr = np.array(lengths)
        return {q: int(np.quantile(arr, q)) for q in quantiles}
