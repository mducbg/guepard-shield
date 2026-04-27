# %% [markdown]
# # Phase 2: Evaluate Syscall Transformer (Window-level)
#
# This notebook loads the best trained model and evaluates its performance
# on the LID-DS-2021 Test set using both Window-level and Recording-level metrics.
# Supports configurable NLL aggregation (mean / max / p95) and FPR-constrained
# threshold selection for deployment-ready metrics.

# %%
from typing import Optional, Dict, List
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)

from gp.config import npy_dir, ckpt_path
from gp.model.transformer import SyscallTransformer


# %% [markdown]
# ## 1. Load Model

# %%
# Teacher checkpoint — explicitly pinned, do not change.
# This is the subsampled model (val_loss=0.3455) selected as Teacher in P2.
# See docs/WALKTHROUGH.md for the model comparison rationale.
best_ckpt = Path("results/checkpoints/transformer/best/best-transformer-epoch=29-val_loss=0.3455.ckpt")
if not best_ckpt.exists():
    raise FileNotFoundError(f"Teacher checkpoint not found: {best_ckpt}")
print(f"Loading Teacher checkpoint: {best_ckpt.name}")

model = SyscallTransformer.load_from_checkpoint(str(best_ckpt))
model.eval()
model.cuda()  # Use GPU for fast inference


# %% [markdown]
# ## 2. Inference and Scoring
#
# We calculate the Negative Log Likelihood (NLL) for each window.
# Aggregation can be:
#   - "mean": average NLL (sensitive to overall deviation)
#   - "max":   maximum NLL (catches short anomalous sub-sequences)
#   - "p95":   95th percentile NLL (robust compromise)

# %%
def find_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Return a threshold such that FPR on normal data is <= target_fpr."""
    normal_scores = y_score[y_true == 0]
    if len(normal_scores) == 0:
        return float(np.min(y_score) - 1.0)
    sorted_ns = np.sort(normal_scores)[::-1]  # descending
    idx = int(np.floor(target_fpr * len(sorted_ns)))
    idx = min(idx, len(sorted_ns) - 1)
    return float(sorted_ns[idx])


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_test_set(
    mini_batch_size: int = 32,
    max_normal_recordings: Optional[int] = 1000,
    aggregation: str = "mean",
):
    test_dir = npy_dir / "test"
    # Target only window data files
    window_files = sorted(list(test_dir.glob("*_windows.npy")))

    # 1. Get all exploit recordings
    exploit_files = [f for f in window_files if "_exploit_" in f.name]

    # 2. Get normal recordings and optionally limit them for speed
    normal_files = [f for f in window_files if "_normal_" in f.name]
    if max_normal_recordings is not None and len(normal_files) > max_normal_recordings:
        # Pick a deterministic subset of normal recordings
        normal_files = normal_files[:max_normal_recordings]

    # Combine
    target_files = sorted(exploit_files + normal_files)

    all_window_scores = []
    all_window_labels = []
    recording_results = []

    print(f"Starting window-level inference on {len(target_files)} recordings...")
    print(f"(Exploits: {len(exploit_files)}, Normals: {len(normal_files)}, "
          f"Mini-batch: {mini_batch_size}, Aggregation: {aggregation})")

    with torch.no_grad():
        for f_win in tqdm(target_files):
            # Load windows and corresponding labels
            windows_np = np.load(f_win).astype(np.int64)
            f_label = f_win.parent / f_win.name.replace("_windows.npy", "_labels.npy")
            if not f_label.exists():
                continue
            window_gt_labels = np.load(f_label).astype(np.int8)

            num_windows = windows_np.shape[0]
            window_scores_list = []

            # Process windows in mini-batches to avoid GPU OOM on long recordings
            for start_idx in range(0, num_windows, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, num_windows)
                x_batch = torch.from_numpy(windows_np[start_idx:end_idx]).cuda()

                # Use the new high-level interface with configurable aggregation
                window_scores_batch = model.compute_anomaly_score(x_batch, aggregation=aggregation)
                window_scores_list.append(window_scores_batch.cpu().numpy())

            # Aggregate all mini-batches for this recording
            window_scores = np.concatenate(window_scores_list)

            # Collect for Window-level metrics
            all_window_scores.extend(window_scores)
            all_window_labels.extend(window_gt_labels)

            # Recording-level: Max aggregator across windows of this recording
            is_exploit_rec = 1 if "_exploit_" in f_win.name else 0
            rec_score = window_scores.max()

            recording_results.append({
                "filename": f_win.name,
                "score": rec_score,
                "label": is_exploit_rec,
            })

    return np.array(all_window_scores), np.array(all_window_labels), recording_results


# %%
# Run Evaluation
if __name__ == "__main__":
    # ---- CONFIGURATION ----
    AGGREGATION = "mean"  # "mean" is the only viable aggregation — see WALKTHROUGH.md
    MAX_NORMAL_RECORDINGS = 1000  # None = all
    MINI_BATCH_SIZE = 32
    FPR_TARGETS = [0.01, 0.05, 0.10]

    # Create output directory
    eval_output_dir = Path("results/evaluation/transformer")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    win_scores, win_labels, rec_results = evaluate_test_set(
        mini_batch_size=MINI_BATCH_SIZE,
        max_normal_recordings=MAX_NORMAL_RECORDINGS,
        aggregation=AGGREGATION,
    )

    # 1. Window-level Metrics
    win_auroc = roc_auc_score(win_labels, win_scores)

    # 2. Recording-level Metrics
    rec_scores = np.array([r['score'] for r in rec_results])
    rec_labels = np.array([r['label'] for r in rec_results])
    rec_auroc = roc_auc_score(rec_labels, rec_scores)

    # 3. Best unconstrained F1 threshold (window-level)
    prec, rec, thresh = precision_recall_curve(win_labels, win_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1)
    best_f1 = f1[best_idx]
    optimal_threshold = thresh[best_idx] if best_idx < len(thresh) else 1.0

    # 4. FPR-constrained thresholds (window-level)
    fpr_results: List[Dict] = []
    for target in FPR_TARGETS:
        t = find_threshold_at_fpr(win_labels, win_scores, target)
        metrics = evaluate_at_threshold(win_labels, win_scores, t)
        fpr_results.append(metrics)

    # 5. FPR-constrained thresholds (recording-level)
    rec_fpr_results: List[Dict] = []
    for target in FPR_TARGETS:
        t = find_threshold_at_fpr(rec_labels, rec_scores, target)
        metrics = evaluate_at_threshold(rec_labels, rec_scores, t)
        rec_fpr_results.append(metrics)

    # Print Results
    print("\n" + "=" * 40)
    print("   DETECTION PERFORMANCE")
    print("=" * 40)
    print(f"Aggregation mode:      {AGGREGATION}")
    print(f"Window-level AUROC:    {win_auroc:.4f}")
    print(f"Recording-level AUROC: {rec_auroc:.4f}")
    print(f"Best Window F1-Score:  {best_f1:.4f} (thr={optimal_threshold:.4f})")
    print("-" * 40)
    print("Window-level @ target FPR:")
    for r in fpr_results:
        print(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): "
              f"Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}")
    print("-" * 40)
    print("Recording-level @ target FPR:")
    for r in rec_fpr_results:
        print(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): "
              f"Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}")
    print("=" * 40)

    # --- SAVE RESULTS ---
    # 1. Save summary text
    with open(eval_output_dir / "summary.txt", "w") as f:
        f.write("=== Transformer Evaluation Summary ===\n")
        f.write(f"Checkpoint: {Path(str(best_ckpt)).name}\n")
        f.write(f"Aggregation: {AGGREGATION}\n")
        f.write(f"Window-level AUROC:    {win_auroc:.4f}\n")
        f.write(f"Recording-level AUROC: {rec_auroc:.4f}\n")
        f.write(f"Best Window F1-Score:  {best_f1:.4f}\n")
        f.write(f"Optimal Threshold:     {optimal_threshold:.4f}\n\n")

        f.write("Window-level @ target FPR:\n")
        for r in fpr_results:
            f.write(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): "
                    f"Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}\n")

        f.write("\nRecording-level @ target FPR:\n")
        for r in rec_fpr_results:
            f.write(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): "
                    f"Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}\n")

    # 2. Save raw window scores for plotting
    np.savez(
        eval_output_dir / "window_scores.npz",
        scores=win_scores,
        labels=win_labels,
        aggregation=AGGREGATION,
    )

    # 3. Save recording-level CSV for detailed analysis
    import pandas as pd
    df_rec = pd.DataFrame(rec_results)
    df_rec.to_csv(eval_output_dir / "recording_predictions.csv", index=False)

    print(f"\nResults saved to: {eval_output_dir}")
