"""DFA evaluation: FPR, per-recording detection rate, and fidelity.

Evaluation is per-window stateless: every window starts from DFA state 0.
A window is REJECTED the moment any token has no outgoing transition.

Primary metric: per-recording detection rate (DR).
  LID-DS labels 98.85% of attack-period windows as "attack" even though the
  server reverts to normal syscall behaviour after the exploit burst.  The DFA
  correctly accepts those post-exploit normal windows, so per-window TPR on raw
  labels is misleading.  The correct question is:
      "Did the DFA reject at least ONE window inside an attack recording?"
  This is DR (detection rate at recording level).

Per-window TPR on raw labels is still computed and stored for completeness /
comparison with the teacher, but it is NOT the primary evaluation metric.
"""

from __future__ import annotations

import numpy as np


class DFAEvaluator:
    """Evaluate a DFA transition table against labelled syscall windows.

    Args:
        K:          Number of DFA states (max state ID + 1).
        vocab_size: Token vocabulary size (including PAD=0).
        chunk_size: Windows processed per vectorised batch.
        start_state: DFA start state (cluster ID of the most common initial hidden state).
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        chunk_size: int = 500_000,
        start_state: int = 0,
    ) -> None:
        self.K = K
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.start_state = start_state

    def evaluate_all(
        self,
        transitions: dict[tuple[int, int], int],
        test_X: np.ndarray,
        test_labels: np.ndarray,
        val_X: np.ndarray,
        teacher_decisions: np.ndarray,
        test_rec_ids: np.ndarray | None = None,
    ) -> tuple[float, float, float, float]:
        """Return (FPR, tpr_window, dr_rec, fidelity) building the transition table once.

        Args:
            transitions:      DFA transition dict {(src_state, token): dst_state}.
            test_X:           Test windows [N_test, W] int tokens.
            test_labels:      Raw LID-DS per-window labels [N_test] {0,1}.
            val_X:            Val windows [N_val, W] int tokens.
            teacher_decisions: Teacher reject decisions on val [N_val] bool.
            test_rec_ids:     Recording ID per test window [N_test] int.
                              Required for per-recording detection rate (dr_rec).
                              If None, dr_rec is returned as NaN.

        Returns:
            fpr        — rejected normal windows / total normal windows.
            tpr_window — rejected attack windows / total attack windows (raw labels,
                         inflated by 98.85% post-exploit normal behaviour; secondary only).
            dr_rec     — attack recordings with ≥1 rejection / total attack recordings
                         (PRIMARY detection metric; NaN when test_rec_ids is None).
            fidelity   — agreement rate between DFA and teacher on val set.
        """
        T = self._build_table(transitions)
        reject_test = self._simulate_with_table(T, test_X)
        reject_val  = self._simulate_with_table(T, val_X)

        normal   = test_labels == 0
        attack   = test_labels == 1
        n_normal = int(normal.sum())
        n_attack = int(attack.sum())

        fpr        = float(reject_test[normal].sum()) / n_normal if n_normal > 0 else float("nan")
        tpr_window = float(reject_test[attack].sum()) / n_attack if n_attack > 0 else float("nan")
        fidelity   = float((reject_val == teacher_decisions).mean())

        dr_rec, _, _ = self._per_recording_dr(reject_test, test_labels, test_rec_ids)

        return fpr, tpr_window, dr_rec, fidelity

    def per_recording_detection_rate(
        self,
        transitions: dict[tuple[int, int], int],
        test_X: np.ndarray,
        test_labels: np.ndarray,
        test_rec_ids: np.ndarray,
    ) -> tuple[float, int, int]:
        """Return (DR, n_detected, n_attack_recs) for attack recordings.

        DR = attack recordings with ≥1 DFA rejection / total attack recordings.
        A recording is an "attack recording" if any of its windows has label=1.
        """
        T = self._build_table(transitions)
        reject = self._simulate_with_table(T, test_X)
        dr, n_det, n_atk = self._per_recording_dr(reject, test_labels, test_rec_ids)
        return dr, n_det, n_atk

    def fpr_tpr(
        self,
        transitions: dict[tuple[int, int], int],
        test_X: np.ndarray,
        test_labels: np.ndarray,
    ) -> tuple[float, float]:
        """Return (FPR, TPR_window) over test windows.

        Note: TPR_window uses raw LID-DS labels which contain 98.85% contamination
        (post-exploit normal windows). Prefer per_recording_detection_rate() instead.
        """
        T = self._build_table(transitions)
        reject = self._simulate_with_table(T, test_X)
        normal = test_labels == 0
        attack = test_labels == 1
        n_normal = int(normal.sum())
        n_attack = int(attack.sum())
        fpr = float(reject[normal].sum()) / n_normal if n_normal > 0 else float("nan")
        tpr = float(reject[attack].sum()) / n_attack if n_attack > 0 else float("nan")
        return fpr, tpr

    def fidelity(
        self,
        transitions: dict[tuple[int, int], int],
        val_X: np.ndarray,
        teacher_decisions: np.ndarray,
    ) -> float:
        """Agreement rate between DFA and teacher on the validation set."""
        T = self._build_table(transitions)
        dfa_decisions = self._simulate_with_table(T, val_X)
        return float((dfa_decisions == teacher_decisions).mean())

    # ── internal ──────────────────────────────────────────────────────────────

    def _per_recording_dr(
        self,
        reject: np.ndarray,
        labels: np.ndarray,
        rec_ids: "np.ndarray | None",
    ) -> "tuple[float, int, int]":
        """Compute per-recording detection rate from a per-window reject mask.

        A recording is an attack recording if any of its windows has label=1.
        DR = # attack recordings with ≥1 rejection / # attack recordings.

        Returns:
            (DR, n_detected, n_attack_recs).  DR is NaN when rec_ids is None.
        """
        if rec_ids is None:
            return float("nan"), 0, 0

        # Segment boundaries by recording ID (standard approach)
        ids = np.asarray(rec_ids)
        starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
        ends   = np.r_[starts[1:], len(ids)]

        n_atk_recs = 0
        n_detected  = 0
        for s, e in zip(starts, ends):
            if not labels[s:e].any():
                continue  # normal recording — skip
            n_atk_recs += 1
            if reject[s:e].any():
                n_detected += 1

        dr = n_detected / n_atk_recs if n_atk_recs > 0 else float("nan")
        return dr, n_detected, n_atk_recs

    def _build_table(self, transitions: dict[tuple[int, int], int]) -> np.ndarray:
        """Return lookup table T[state, token] = next_state (-1 = missing)."""
        T = np.full((self.K, self.vocab_size), -1, dtype=np.int32)
        for (src, tok), dst in transitions.items():
            if 0 <= src < self.K and 0 <= tok < self.vocab_size and 0 <= dst < self.K:
                T[src, tok] = dst
        return T

    def _simulate_with_table(self, T: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Simulate DFA on all windows using a prebuilt table; return bool reject array [N].

        Vectorised over the batch dimension.
        """
        N, W = X.shape
        reject = np.zeros(N, dtype=bool)

        for chunk_start in range(0, N, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, N)
            X_chunk = np.asarray(X[chunk_start:chunk_end])
            n = chunk_end - chunk_start

            states = np.full(n, self.start_state, dtype=np.int32)
            rej = np.zeros(n, dtype=bool)

            for w in range(W):
                tok = X_chunk[:, w]           # [n] token IDs
                pad = tok == 0                # PAD token — stop advancing

                # Safe lookup: use state=0 for already-rejected windows
                safe_states = np.where(rej, 0, states)
                safe_tok = np.clip(tok, 0, self.vocab_size - 1)
                next_st = T[safe_states, safe_tok]  # [n]

                # Reject if transition missing (and not already done, not PAD)
                new_rej = (next_st < 0) & ~rej & ~pad
                rej |= new_rej

                # Advance state only for live, non-PAD windows
                advance = ~rej & ~pad
                states = np.where(advance, next_st, states)

            reject[chunk_start:chunk_end] = rej

        return reject
