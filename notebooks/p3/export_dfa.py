"""Phase 3 — Step 5: Export the best DFA config to dfa_config.json.

Run from project root:
    uv run notebooks/p3/export_dfa.py

Selection criterion: lowest FPR with TPR ≥ 0.5.
Reads grid_search.csv produced by eval_dfa.py.

Output:
    results/p3/dfa_config.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gp.dfa.export import DFAExporter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

HIDDEN_DIR  = PROJECT_ROOT / "results" / "p3" / "hidden_states"
CLUSTER_DIR = PROJECT_ROOT / "results" / "p3" / "clusters"
DFA_DIR     = PROJECT_ROOT / "results" / "p3" / "dfa"
METRICS_DIR = PROJECT_ROOT / "results" / "p3" / "metrics"
VOCAB_PATH  = PROJECT_ROOT / "data" / "processed" / "p2" / "vocab.json"
OUT_PATH    = PROJECT_ROOT / "results" / "p3" / "dfa_config.json"

MIN_TPR = 0.5


def read_grid(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def select_best(rows: list[dict]) -> dict | None:
    """Return the row with minimum FPR subject to TPR ≥ MIN_TPR.

    S1 configs are excluded: after subset construction their state IDs are
    subset-construction IDs, not K-Means cluster IDs, so state_tiers computed
    from cluster frequencies would be wrong.
    """
    s1_excluded = [r for r in rows if r["strategy"] == "S1"]
    if s1_excluded:
        print(
            f"  [note] Excluding {len(s1_excluded)} S1 config(s) from export "
            f"candidates (subset-construction state IDs incompatible with "
            f"cluster-frequency state_tiers)."
        )
    candidates = [
        r for r in rows
        if float(r["tpr"]) >= MIN_TPR and r["strategy"] != "S1"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["fpr"]))


def load_transitions(config_name: str) -> dict[tuple[int, int], int]:
    dfa_dir = DFA_DIR / config_name
    data = np.load(dfa_dir / "transitions.npz")
    src, tok, dst = data["src"], data["tok"], data["dst"]
    return {(int(s), int(t)): int(d) for s, t, d in zip(src, tok, dst)}


def compute_state_freqs(config_name: str, M: int) -> np.ndarray:
    """Count how many training windows land in each cluster state."""
    parts = config_name.split("_")
    K = int(parts[0][1:])
    labels_path = CLUSTER_DIR / f"K{K}" / "labels.dat"
    labels = np.memmap(labels_path, dtype="int32", mode="r", shape=(M,))
    freqs = np.bincount(np.asarray(labels), minlength=K).astype(np.int64)
    return freqs


def main() -> None:
    csv_path = METRICS_DIR / "grid_search.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run eval_dfa.py first ({csv_path} missing)")

    rows = read_grid(csv_path)
    best = select_best(rows)
    if best is None:
        print(f"No config found with TPR ≥ {MIN_TPR}. Best configs by TPR:")
        for r in sorted(rows, key=lambda r: float(r["tpr"]), reverse=True)[:5]:
            print(f"  {r['config']}: TPR={r['tpr']}  FPR={r['fpr']}")
        return

    config_name = best["config"]
    print(f"Best config: {config_name}")
    print(f"  FPR={best['fpr']}  TPR={best['tpr']}  fidelity={best['fidelity']}")
    print(f"  n_states={best['n_states']}  n_trans={best['n_trans']}")

    with open(HIDDEN_DIR / "info.json") as f:
        info = json.load(f)
    M = info["M"]

    parts = config_name.split("_")
    K = int(parts[0][1:])
    strategy = parts[1]
    theta_str = parts[2][1:] if len(parts) > 2 else ""

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    transitions = load_transitions(config_name)
    state_freqs = compute_state_freqs(config_name, M)

    metadata = {
        "K": K,
        "strategy": strategy,
        "theta": float(theta_str) if theta_str else None,
        "vocab_size": info["vocab_size"],
        "n_states": int(best["n_states"]),
        "n_transitions": int(best["n_trans"]),
        "fpr": float(best["fpr"]),
        "tpr": float(best["tpr"]),
        "fidelity": float(best["fidelity"]),
    }

    exporter = DFAExporter()
    config = exporter.to_json(
        transitions=transitions,
        vocab=vocab,
        state_freqs=state_freqs,
        metadata=metadata,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nExported → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    try:
        plot_best_dfa_stats(config_name, transitions, state_freqs)
    except Exception as e:
        print(f"Error plotting best DFA stats: {e}")


def plot_best_dfa_stats(
    config_name: str,
    transitions: dict[tuple[int, int], int],
    state_freqs: np.ndarray,
) -> None:
    plots_dir = PROJECT_ROOT / "results" / "p3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    out_degrees = {}
    for (src, tok), dst in transitions.items():
        out_degrees[src] = out_degrees.get(src, 0) + 1
    for s in range(len(state_freqs)):
        if s not in out_degrees:
            out_degrees[s] = 0
            
    degrees = list(out_degrees.values())
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(degrees, bins=range(min(degrees), max(degrees) + 2), kde=False, color='dodgerblue')
    plt.xlabel("Out-Degree (Number of outgoing transitions)")
    plt.ylabel("Number of States")
    plt.title(f"State Out-Degree Distribution ({config_name})")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_dfa_out_degrees.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sorted_freqs = np.sort(state_freqs)[::-1]
    total = sum(state_freqs)
    pct = (sorted_freqs / total) * 100 if total > 0 else sorted_freqs
    plt.plot(np.arange(1, len(pct) + 1), pct, marker='o', color='forestgreen', markersize=4)
    plt.xlabel("State Rank (sorted by frequency)")
    plt.ylabel("Percentage of training windows (%)")
    plt.title(f"State Frequency Distribution ({config_name})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_dfa_state_frequencies.png", dpi=300)
    plt.close()
    
    print(f"Saved best DFA stats plots to {plots_dir}")


if __name__ == "__main__":
    main()
