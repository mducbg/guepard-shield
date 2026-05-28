"""Phase 3 — Step 4: Grid-search all DFA configs for FPR, DR, fidelity.

Run from project root:
    uv run notebooks/p3/eval_dfa.py \\
        [--ckpt results/p2/checkpoints/best.ckpt]

For each (K, strategy) config:
    - Simulates DFA on test windows → FPR, TPR_window, DR_rec
    - Simulates DFA on val windows vs. teacher → fidelity

Primary metrics:
  - DR_rec (detection rate at recording level): fraction of attack recordings
    where the DFA rejects ≥1 window.  This is the correct metric because
    98.85% of LID-DS labeled-attack windows are post-exploit normal behaviour
    that the DFA SHOULD accept — per-window TPR on raw labels is misleading.
  - FPR: fraction of normal windows incorrectly rejected.

Secondary / diagnostic metrics:
  - TPR_window: per-window rejection rate on raw LID-DS attack labels.
    Reported for completeness but NOT the primary quality indicator.
  - Fidelity: agreement rate with teacher on val set.
    A DFA that accepts everything achieves ~VAL_TAU_PCT% fidelity by default;
    use only for comparing configs, not as an absolute quality score.

Evaluation is per-window cold-start (every window restarts from the start
state). The eBPF deployment maintains streaming per-thread state, so cold-start
FPR may overestimate real-world FPR for mid-recording normal windows.

Teacher val scores are cached at results/p3/val_teacher_scores.npy on first run.

Output:
    results/p3/metrics/grid_search.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from gp.datamodule import SyscallDataModule
from gp.dfa.evaluate import DFAEvaluator
from gp.model import SyscallTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "p2"
HIDDEN_DIR   = PROJECT_ROOT / "results" / "p3" / "hidden_states"
CLUSTER_DIR  = PROJECT_ROOT / "results" / "p3" / "clusters"
DFA_DIR      = PROJECT_ROOT / "results" / "p3" / "dfa_s1"
METRICS_DIR  = PROJECT_ROOT / "results" / "p3" / "metrics"
VAL_SCORES_PATH = PROJECT_ROOT / "results" / "p3" / "val_teacher_scores.npy"

K_VALUES   = [64, 128, 256, 512]
S4_THETAS  = [0.80, 0.90, 0.95, 0.99]
VAL_TAU_PCT = 99.5


def config_dirs() -> list[tuple[str, Path]]:
    """Enumerate all (config_name, path) pairs that have a transitions.npz."""
    configs = []
    for K in K_VALUES:
        for tag in ["S1", "S3"] + [f"S4_t{t:.2f}" for t in S4_THETAS]:
            d = DFA_DIR / f"K{K}_{tag}"
            npz = d / "transitions.npz"
            if npz.exists():
                configs.append((f"K{K}_{tag}", d))
    return configs


def load_transitions(dfa_dir: Path) -> dict[tuple[int, int], int] | None:
    """Load transitions.npz → dict. Returns None for state-explosion configs."""
    if (dfa_dir / "state_explosion.txt").exists():
        return None
    npz = dfa_dir / "transitions.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    src, tok, dst = data["src"], data["tok"], data["dst"]
    return {(int(s), int(t)): int(d) for s, t, d in zip(src, tok, dst)}


def cache_val_teacher_scores(ckpt: str, batch_size: int) -> np.ndarray:
    """Run teacher inference on val set and cache NLL scores."""
    if VAL_SCORES_PATH.exists():
        print(f"Loading cached val teacher scores from {VAL_SCORES_PATH}")
        return np.load(VAL_SCORES_PATH)

    print("Running teacher inference on val set (one-time, will be cached)...")
    model = SyscallTransformer.load_from_checkpoint(ckpt)
    from lightning import Trainer

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision="16-mixed",
    )
    dm = SyscallDataModule(DATA_DIR, batch_size=batch_size, num_workers=4)
    dm.setup("fit")

    predictions = trainer.predict(model, dataloaders=dm.val_dataloader())
    last_scores = np.concatenate([p[0].numpy() for p in predictions]).astype(np.float32)

    VAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(VAL_SCORES_PATH, last_scores)
    print(f"Cached val scores → {VAL_SCORES_PATH}  (N={len(last_scores):,})")
    return last_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(PROJECT_ROOT / "results" / "p2" / "checkpoints" / "best.ckpt"),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    info_path = HIDDEN_DIR / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Run extract_states.py first ({info_path} missing)")

    with open(info_path) as f:
        info = json.load(f)
    vocab_size = info["vocab_size"]

    # Load test set (mmap — 24GB, never fully in RAM)
    print("Loading test set (mmap)...")
    test_X      = np.load(DATA_DIR / "test_X.npy",       mmap_mode="r")
    test_y      = np.load(DATA_DIR / "test_y.npy",       mmap_mode="r")
    test_rec_ids_path = DATA_DIR / "test_rec_ids.npy"
    test_rec_ids = (
        np.load(test_rec_ids_path, mmap_mode="r")
        if test_rec_ids_path.exists()
        else None
    )
    if test_rec_ids is None:
        print("  WARNING: test_rec_ids.npy not found — per-recording DR will be NaN")
    print(f"  test: {len(test_X):,} windows  ({int(test_y.sum()):,} attack windows)")

    # Load val set (1.3 GB — fits in RAM)
    print("Loading val set...")
    val_X = np.load(DATA_DIR / "val_X.npy", mmap_mode="r")

    # Cache teacher val scores
    val_scores = cache_val_teacher_scores(args.ckpt, args.batch_size)
    oracle_tau = float(np.percentile(val_scores, VAL_TAU_PCT))
    teacher_decisions = val_scores > oracle_tau
    print(f"  oracle_tau (p{VAL_TAU_PCT}) = {oracle_tau:.4f}")
    print(f"  teacher rejects {teacher_decisions.mean():.2%} of val windows")

    # Grid search
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "grid_search.csv"
    # dr_rec is the PRIMARY metric (per-recording detection rate).
    # tpr_window is secondary only — raw labels contain 98.85% contamination.
    fieldnames = ["config", "K", "strategy", "theta",
                  "fpr", "dr_rec", "tpr_window",
                  "fidelity", "n_states", "n_trans"]

    configs = config_dirs()
    print(f"\nEvaluating {len(configs)} configs...")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config_name, dfa_dir in tqdm(configs, desc="grid search"):
            transitions = load_transitions(dfa_dir)
            if transitions is None:
                print(f"  [skip] {config_name}: state explosion or missing")
                continue

            # Parse config name: K{k}_{S}[_{tag}]
            parts = config_name.split("_")
            K = int(parts[0][1:])
            strategy = parts[1]
            theta = float(parts[2][1:]) if len(parts) > 2 else float("nan")

            # Read pre-saved n_states from npz (avoids crash when transitions is empty)
            meta_npz = np.load(dfa_dir / "transitions.npz")
            n_states_trans = int(meta_npz["n_states"])
            n_trans = len(transitions)

            # Load per-K start state (most common initial cluster in training data)
            for _ss_name in ("start_state_s1.txt", "start_state.txt"):
                _ss_path = CLUSTER_DIR / f"K{K}" / _ss_name
                if _ss_path.exists():
                    start_state = int(_ss_path.read_text().strip())
                    break
            else:
                start_state = 0
            # S1 DFA state 0 maps to NFA {start_cluster}; use 0 as S1 DFA start.
            dfa_start = 0 if strategy == "S1" else start_state

            evaluator = DFAEvaluator(
                K=n_states_trans, vocab_size=vocab_size, start_state=dfa_start
            )
            fpr, tpr_window, dr_rec, fid = evaluator.evaluate_all(
                transitions, test_X, test_y, val_X, teacher_decisions,
                test_rec_ids=test_rec_ids,
            )

            dr_str  = f"{dr_rec:.6f}" if not (isinstance(dr_rec, float) and dr_rec != dr_rec) else "nan"
            row = {
                "config":     config_name,
                "K":          K,
                "strategy":   strategy,
                "theta":      "" if np.isnan(theta) else f"{theta:.2f}",
                "fpr":        f"{fpr:.6f}",
                "dr_rec":     dr_str,
                "tpr_window": f"{tpr_window:.6f}",
                "fidelity":   f"{fid:.6f}",
                "n_states":   n_states_trans,
                "n_trans":    n_trans,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"  {config_name:<20s}  FPR={fpr:.4f}  DR={dr_rec:.4f}"
                f"  TPR_win={tpr_window:.4f}  fidelity={fid:.4f}"
                f"  states={n_states_trans}  trans={n_trans}"
            )

    print(f"\nGrid search complete → {csv_path}")
    try:
        plot_grid_results(csv_path)
    except Exception as e:
        print(f"Error plotting grid search results: {e}")


def plot_grid_results(csv_path: Path) -> None:
    plots_dir = PROJECT_ROOT / "results" / "p3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty grid search CSV. Skipping plots.")
        return
        
    df['fpr']        = df['fpr'].astype(float)
    df['dr_rec']     = pd.to_numeric(df['dr_rec'], errors='coerce')
    df['tpr_window'] = df['tpr_window'].astype(float)
    df['fidelity']   = df['fidelity'].astype(float)
    df['K']          = df['K'].astype(int)
    df['n_states']   = df['n_states'].astype(int)
    df['n_trans']    = df['n_trans'].astype(int)
    
    def get_label(row):
        if row['strategy'] == 'S4':
            try:
                val = float(row['theta'])
                return f"S4 (θ={val:.2f})"
            except Exception:
                return f"S4 (θ={row['theta']})"
        return row['strategy']
        
    df['strategy_label'] = df.apply(get_label, axis=1)
    
    sns.set_theme(style="whitegrid")
    
    # 1. DR_rec vs FPR Scatter (PRIMARY metric plot)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x='fpr',
        y='dr_rec',
        hue='strategy_label',
        style='K',
        s=120,
        alpha=0.85
    )
    plt.xlabel("False Positive Rate (FPR) — rejected normal windows / total normal")
    plt.ylabel("Detection Rate DR (per recording) — primary metric")
    plt.title(
        "DFA Grid Search: Detection Rate vs. FPR\n"
        "DR = attack recordings with ≥1 rejection / total attack recordings\n"
        "(98.85% of LID-DS attack-window labels are post-exploit normal — DR is the correct metric)"
    )
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "dfa_dr_vs_fpr.png", dpi=300)
    plt.close()

    # 1b. TPR_window vs FPR (secondary / diagnostic only)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x='fpr',
        y='tpr_window',
        hue='strategy_label',
        style='K',
        s=120,
        alpha=0.85
    )
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("TPR (per window, raw labels) — secondary / diagnostic only")
    plt.title(
        "DFA Grid Search: TPR_window vs. FPR  [DIAGNOSTIC ONLY]\n"
        "Raw per-window TPR is inflated by 98.85% post-exploit normal windows.\n"
        "Use DR_rec (dfa_dr_vs_fpr.png) as the primary metric."
    )
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "dfa_tpr_window_vs_fpr.png", dpi=300)
    plt.close()
    
    # 2. Fidelity vs K Lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='K',
        y='fidelity',
        hue='strategy_label',
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Fidelity vs. Teacher Model")
    plt.title("DFA Fidelity vs. Number of Clusters (K)")
    plt.xticks(sorted(df['K'].unique()))
    plt.tight_layout()
    plt.savefig(plots_dir / "dfa_fidelity_vs_k.png", dpi=300)
    plt.close()
    
    # 3. Transitions vs States Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='n_states',
        y='n_trans',
        hue='strategy_label',
        style='K',
        s=120,
        alpha=0.85
    )
    plt.xlabel("Number of States in DFA")
    plt.ylabel("Number of Transitions")
    plt.title("State Space Complexity: Transitions vs. States")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "dfa_states_vs_transitions.png", dpi=300)
    plt.close()
    
    print(f"Saved eval plots to {plots_dir}")


if __name__ == "__main__":
    main()
