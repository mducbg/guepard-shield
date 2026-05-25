"""Phase 3 — Step 3: Build NFA/DFA transitions for all (K, strategy) configs.

Run from project root:
    uv run notebooks/p3/build_dfa.py

For each K ∈ {64, 128, 256, 512} and strategy ∈ {S1, S3, S4(θ)}:
  - Builds NFA from consecutive hidden-state cluster pairs
  - Reports nd_rate diagnostic
  - Resolves to DFA and saves transitions.npz

Outputs (results/p3/dfa/K{k}_{strategy}/):
    transitions.npz   — src, tok, dst arrays + n_states, n_trans
    nd_rate.txt       — non-determinism diagnostic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

from gp.dfa.transitions import TransitionBuilder

HIDDEN_DIR  = PROJECT_ROOT / "results" / "p3" / "hidden_states"
CLUSTER_DIR = PROJECT_ROOT / "results" / "p3" / "clusters"
DFA_DIR     = PROJECT_ROOT / "results" / "p3" / "dfa"

K_VALUES = [64, 128, 256, 512]
S4_THETAS = [0.80, 0.90, 0.95, 0.99]


def load_info() -> dict:
    with open(HIDDEN_DIR / "info.json") as f:
        return json.load(f)


def save_transitions(
    path: Path,
    transitions: dict[tuple[int, int], int],
    n_states: int,
) -> None:
    if not transitions:
        src_arr = np.empty(0, dtype=np.int32)
        tok_arr = np.empty(0, dtype=np.int32)
        dst_arr = np.empty(0, dtype=np.int32)
    else:
        keys = list(transitions.keys())
        src_arr = np.array([k[0] for k in keys], dtype=np.int32)
        tok_arr = np.array([k[1] for k in keys], dtype=np.int32)
        dst_arr = np.array(list(transitions.values()), dtype=np.int32)

    np.savez(
        path,
        src=src_arr,
        tok=tok_arr,
        dst=dst_arr,
        n_states=np.int32(n_states),
        n_trans=np.int32(len(transitions)),
    )


def run_k(K: int, M: int, vocab_size: int, stride: int = 1) -> None:
    cluster_dir = CLUSTER_DIR / f"K{K}"
    labels_path   = cluster_dir / "labels.dat"
    if not labels_path.exists():
        print(f"  [skip] K={K}: labels.dat not found, run cluster.py first")
        return

    print(f"\n══ K={K} ════════════════════════════════════════════")

    labels = np.memmap(labels_path, dtype="int32", mode="r", shape=(M,))
    meta   = np.memmap(
        HIDDEN_DIR / "train_meta.dat", dtype="int32", mode="r", shape=(M, 3)
    )

    builder = TransitionBuilder(labels, meta, vocab_size, stride=stride)
    print(f"  Building NFA...")
    builder.build_nfa()

    nd = builder.nd_rate()
    print(f"  nd_rate = {nd:.1%}")

    # Determine the initial cluster: most common cluster for the first window of
    # each recording (pos_in_rec == 0).  This is the correct DFA start state.
    initial_mask = np.asarray(meta[:, 1]) == 0
    initial_labels = np.asarray(labels)[initial_mask]
    start_cluster = int(np.bincount(initial_labels, minlength=K).argmax())
    (cluster_dir / "start_state.txt").write_text(str(start_cluster) + "\n")
    print(f"  start_cluster = {start_cluster}")

    # ── S1 ─────────────────────────────────────────────────────────────────
    out = DFA_DIR / f"K{K}_S1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nd_rate.txt").write_text(f"{nd:.6f}\n")

    print("  Resolving S1 (subset construction)...")
    s1_trans = builder.resolve_s1(initial_cluster=start_cluster)
    if s1_trans is None:
        print(f"  S1: state explosion — skipping export")
        (out / "transitions.npz").unlink(missing_ok=True)
        (out / "state_explosion.txt").write_text("1\n")
    else:
        n_s1 = max((max(s, d) for (s, _), d in s1_trans.items()), default=-1) + 1
        save_transitions(out / "transitions", s1_trans, n_s1)
        (out / "state_explosion.txt").unlink(missing_ok=True)
        print(f"  S1: {n_s1} states, {len(s1_trans)} transitions")

    # ── S3 ─────────────────────────────────────────────────────────────────
    out = DFA_DIR / f"K{K}_S3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nd_rate.txt").write_text(f"{nd:.6f}\n")

    print("  Resolving S3 (majority vote)...")
    s3_trans = builder.resolve_s3()
    save_transitions(out / "transitions", s3_trans, K)
    print(f"  S3: {K} states, {len(s3_trans)} transitions")

    # ── S4 ─────────────────────────────────────────────────────────────────
    for theta in S4_THETAS:
        tag = f"t{theta:.2f}"
        out = DFA_DIR / f"K{K}_S4_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "nd_rate.txt").write_text(f"{nd:.6f}\n")

        s4_trans = builder.resolve_s4(theta)
        save_transitions(out / "transitions", s4_trans, K)
        print(f"  S4(θ={theta:.2f}): {K} states, {len(s4_trans)} transitions")


def plot_nd_rate() -> None:
    plots_dir = PROJECT_ROOT / "results" / "p3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    k_vals = []
    nd_rates = []
    
    for K in K_VALUES:
        nd_file = DFA_DIR / f"K{K}_S3" / "nd_rate.txt"
        if nd_file.exists():
            try:
                val = float(nd_file.read_text().strip())
                k_vals.append(K)
                nd_rates.append(val)
            except Exception:
                pass
                
    if not nd_rates:
        print("No nd_rate data found to plot.")
        return
        
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, nd_rates, marker='o', linewidth=2, color='crimson')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("NFA Non-Determinism Rate")
    plt.title("Non-Determinism Rate vs. Number of Clusters")
    plt.xticks(K_VALUES)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.tight_layout()
    plot_path = plots_dir / "nd_rate_vs_k.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved non-determinism rate plot → {plot_path}")


def main() -> None:
    info = load_info()
    M = info["M"]
    vocab_size = info["vocab_size"]
    stride = info.get("stride", 1)

    DFA_DIR.mkdir(parents=True, exist_ok=True)

    for K in K_VALUES:
        run_k(K, M, vocab_size, stride=stride)

    print("\nAll DFA configs built.")
    try:
        plot_nd_rate()
    except Exception as e:
        print(f"Error plotting non-determinism rate: {e}")


if __name__ == "__main__":
    main()
