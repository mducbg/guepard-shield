"""Export decision set rules to Rust/Aya-compatible JSON config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from .decision_set import GreedyDecisionSet


class RustConfigExporter:
    """
    Export rules to JSON config for ingestion by the Rust/Aya eBPF pipeline.

    Schema:
      vocab: list of syscall names (index = syscall ID)
      dangerous_syscalls: names used for the dangerous_rate feature
      window_size: window length in syscalls
      rules: list of {feature_name, feature_idx, operator, threshold, precision, support}
    """

    def __init__(self, vocab: List[str], dangerous_syscalls: List[str], window_size: int = 1000):
        self.vocab = vocab
        self.dangerous_syscalls = dangerous_syscalls
        self.window_size = window_size

    def export(self, decision_set: GreedyDecisionSet, output_path: Path) -> None:
        rules: List[Dict[str, Any]] = [
            {
                "feature_name": r.feature_name,
                "feature_idx": r.feature_idx,
                "operator": r.operator,
                "threshold": float(r.threshold),
                "precision": float(r.precision),
                "support": r.support,
            }
            for r in decision_set.rules
        ]

        config = {
            "vocab": self.vocab,
            "dangerous_syscalls": self.dangerous_syscalls,
            "window_size": self.window_size,
            "n_rules": len(rules),
            "rules": rules,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Exported {len(rules)} rules to: {output_path}")
        print(f"  Vocab size: {len(self.vocab)}")
