"""DFA export: serialize a resolved transition table to dfa_config.json.

The JSON format is designed for direct loading into eBPF maps at Phase 4:
    transitions[state][token] = next_state
    Missing entries → REJECT at runtime.

Cluster centroids are NOT serialized here; Phase 4 reads centroids.npy directly.
"""

from __future__ import annotations

import numpy as np


class DFAExporter:
    """Serialize a DFA transition dict to the dfa_config.json schema."""

    def to_json(
        self,
        transitions: dict[tuple[int, int], int],
        vocab: dict[str, int],
        state_freqs: np.ndarray,
        edge_percentile: int = 5,
        metadata: dict | None = None,
    ) -> dict:
        """Build the dfa_config dict ready for json.dump().

        Args:
            transitions:     {(src, token): next_state} DFA table.
            vocab:           {syscall_name: token_id} mapping.
            state_freqs:     [K] int — number of training windows in each state.
            edge_percentile: States below this freq percentile → EDGE tier.
            metadata:        Free-form dict merged into the "metadata" block.
        """
        if metadata is None:
            metadata = {}

        # State tier labelling — only EDGE states listed; NORMAL is default.
        edge_thresh = float(np.percentile(state_freqs, edge_percentile))
        state_tiers = {
            str(s): "EDGE"
            for s, freq in enumerate(state_freqs)
            if float(freq) <= edge_thresh
        }

        # Build nested transitions: {str(state): {str(token): next_state}}
        trans_dict: dict[str, dict[str, int]] = {}
        for (src, tok), dst in transitions.items():
            src_key = str(src)
            if src_key not in trans_dict:
                trans_dict[src_key] = {}
            trans_dict[src_key][str(tok)] = int(dst)

        return {
            "metadata": metadata,
            "vocab": vocab,
            "state_tiers": state_tiers,
            "transitions": trans_dict,
        }
