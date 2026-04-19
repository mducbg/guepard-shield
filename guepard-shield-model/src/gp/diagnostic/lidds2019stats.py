"""Stats accumulator for LID-DS-2019 EDA.

LID-DS-2019 has no predefined split, so all accumulators are keyed by
label ("normal" | "exploit") rather than split.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from gp.data_loader.lidds_2019 import LIDDS2019Recording

_LABELS = ("normal", "exploit")


class LIDDS2019Stats:
    def __init__(self) -> None:
        self._seq_lengths: dict[str, list[int]] = defaultdict(list)
        self._thread_counts: dict[str, list[int]] = defaultdict(list)
        self._per_thread_lengths: dict[str, list[int]] = defaultdict(list)
        self._vocab_freq: dict[str, Counter[str]] = defaultdict(Counter)
        self._recording_counts: dict[str, int] = {lbl: 0 for lbl in _LABELS}
        self._scenario_counts: Counter[str] = Counter()

        # exploit recordings only
        self._attack_offsets: list[float] = []    # exploit_start - warmup (seconds)
        self._attack_fractions: list[float] = []  # offset / (recording_time - warmup)

    def analyze(self, rec: LIDDS2019Recording) -> None:
        label = "exploit" if rec.is_exploit else "normal"
        syscalls = rec.syscalls

        self._recording_counts[label] += 1
        self._scenario_counts[rec.scenario] += 1
        self._seq_lengths[label].append(len(syscalls))

        per_thread: dict[int, int] = defaultdict(int)
        for sc in syscalls:
            per_thread[sc.thread_id] += 1
        self._thread_counts[label].append(len(per_thread))
        for count in per_thread.values():
            self._per_thread_lengths[label].append(count)

        for sc in syscalls:
            self._vocab_freq[label][sc.syscall] += 1

        if rec.is_exploit and rec.exploit_start_time >= 0:
            offset = rec.exploit_start_time - rec.warmup_time
            self._attack_offsets.append(offset)
            post_warmup = rec.recording_time - rec.warmup_time
            if post_warmup > 0:
                self._attack_fractions.append(offset / post_warmup)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> dict[str, int]:
        return {lbl: len(ctr) for lbl, ctr in self._vocab_freq.items()}

    @property
    def oov_exploit(self) -> set[str]:
        """Syscalls seen in exploit recordings but not in normal ones."""
        return set(self._vocab_freq.get("exploit", {})) - set(self._vocab_freq.get("normal", {}))

    def seq_length_percentiles(
        self, label: str, quantiles: list[float]
    ) -> dict[float, int]:
        lengths = self._seq_lengths[label]
        if not lengths:
            return {q: 0 for q in quantiles}
        arr = np.array(lengths)
        return {q: int(np.quantile(arr, q)) for q in quantiles}
