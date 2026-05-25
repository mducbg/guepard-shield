"""DFA transition building from consecutive hidden-state cluster pairs.

Phase 3 workflow:
    labels [M] + meta [M, 3] → NFA → resolve → DFA transitions dict
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

_CHUNK = 100_000  # rows per iteration when scanning M-length arrays


class TransitionBuilder:
    """Build NFA from strided hidden-state pairs and resolve to a DFA.

    Args:
        labels:     [M] int array — cluster ID per window.
        meta:       [M, 3] int array — (rec_id, pos_in_rec, token_id) per window.
        vocab_size: number of syscall tokens (including PAD/UNK).
        stride:     extraction stride used in extract_states.py (default 1).
                    Adjacent windows satisfy pos[i+1] == pos[i] + stride.
    """

    def __init__(
        self,
        labels: np.ndarray,
        meta: np.ndarray,
        vocab_size: int,
        stride: int = 1,
    ) -> None:
        self.labels = labels
        self.meta = meta
        self.vocab_size = vocab_size
        self.stride = stride
        self._nfa: dict[tuple[int, int], Counter[int]] | None = None

    @property
    def K(self) -> int:
        return int(self.labels.max()) + 1

    # ── NFA building ──────────────────────────────────────────────────────────

    def build_nfa(self) -> dict[tuple[int, int], Counter[int]]:
        """Build NFA transition counts from consecutive strided window pairs.

        Pair (i, i+1) is valid iff same rec_id AND pos[i+1] == pos[i] + stride.
        Transition: src=labels[i], token=meta[i+1,2], dst=labels[i+1].

        Returns:
            nfa[(src, token)] = Counter{dst: count}
        """
        nfa: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        M = len(self.labels)

        for start in range(0, M - 1, _CHUNK):
            end = min(start + _CHUNK + 1, M)
            chunk_meta = np.asarray(self.meta[start:end])
            chunk_labels = np.asarray(self.labels[start:end])

            same_rec = chunk_meta[1:, 0] == chunk_meta[:-1, 0]
            adj_pos = chunk_meta[1:, 1] == chunk_meta[:-1, 1] + self.stride
            mask = same_rec & adj_pos

            src_arr = chunk_labels[:-1][mask]
            tok_arr = chunk_meta[1:, 2][mask]
            dst_arr = chunk_labels[1:][mask]

            if len(src_arr) == 0:
                continue
            pairs = np.stack([src_arr, tok_arr, dst_arr], axis=1)
            unique_triples, counts = np.unique(pairs, axis=0, return_counts=True)
            for (s, t, d), c in zip(unique_triples.tolist(), counts.tolist()):
                nfa[(s, t)][d] += c

        self._nfa = dict(nfa)
        return self._nfa

    def nd_rate(self) -> float:
        """Fraction of (src, token) pairs that have more than 1 unique destination."""
        if self._nfa is None:
            raise RuntimeError("Call build_nfa() first.")
        if not self._nfa:
            return 0.0
        nd = sum(1 for c in self._nfa.values() if len(c) > 1)
        return nd / len(self._nfa)

    # ── Resolution strategies ─────────────────────────────────────────────────

    def resolve_s1(
        self, initial_cluster: int = 0
    ) -> dict[tuple[int, int], int] | None:
        """Subset construction NFA → DFA.

        Returns None if |DFA states| > 10 × K (state explosion).
        DFA state 0 corresponds to the NFA state {initial_cluster}.

        Args:
            initial_cluster: K-Means cluster ID used as the DFA start state.
        """
        if self._nfa is None:
            raise RuntimeError("Call build_nfa() first.")

        limit = 10 * self.K
        nfa = self._nfa

        # Pre-index NFA by source state for fast lookup
        src_to_tokens: dict[int, set[int]] = defaultdict(set)
        for src, tok in nfa:
            src_to_tokens[src].add(tok)

        initial = frozenset({initial_cluster})
        dfa_ids: dict[frozenset, int] = {initial: 0}
        worklist: list[frozenset] = [initial]
        transitions: dict[tuple[int, int], int] = {}

        while worklist:
            if len(dfa_ids) > limit:
                logger.warning(
                    "S1 state explosion: |states|=%d > 10×K=%d",
                    len(dfa_ids),
                    self.K,
                )
                return None

            current = worklist.pop()
            cur_id = dfa_ids[current]

            tokens_seen: set[int] = set()
            for nfa_state in current:
                tokens_seen.update(src_to_tokens.get(nfa_state, set()))

            for tok in tokens_seen:
                next_nfa: set[int] = set()
                for nfa_state in current:
                    key = (nfa_state, tok)
                    if key in nfa:
                        next_nfa.update(nfa[key].keys())

                if not next_nfa:
                    continue

                nxt = frozenset(next_nfa)
                if nxt not in dfa_ids:
                    dfa_ids[nxt] = len(dfa_ids)
                    worklist.append(nxt)

                transitions[(cur_id, tok)] = dfa_ids[nxt]

        return transitions

    def resolve_s3(self) -> dict[tuple[int, int], int]:
        """Majority voting: keep the most frequent destination per (src, token)."""
        if self._nfa is None:
            raise RuntimeError("Call build_nfa() first.")
        return {key: c.most_common(1)[0][0] for key, c in self._nfa.items()}

    def resolve_s4(self, theta: float) -> dict[tuple[int, int], int]:
        """Statistical pruning: keep (src, token) only if dominant branch ≥ theta."""
        if self._nfa is None:
            raise RuntimeError("Call build_nfa() first.")
        result: dict[tuple[int, int], int] = {}
        for key, c in self._nfa.items():
            total = sum(c.values())
            best_dst, best_cnt = c.most_common(1)[0]
            if best_cnt / total >= theta:
                result[key] = best_dst
        return result
