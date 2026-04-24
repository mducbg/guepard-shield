#!/usr/bin/env python
"""
Quick Evaluation Script for Syscall Transformer.

Runs a fast subset evaluation to test aggregation strategies or new checkpoints
without the full 9-hour eval loop.

Usage:
    uv run python notebooks/p2/quick_eval.py --aggregation mean --max-normal 200
    uv run python notebooks/p2/quick_eval.py --aggregation max --max-normal 500
    uv run python notebooks/p2/quick_eval.py --aggregation p95 --max-normal 1000 --batch-size 256
"""

import argparse
import os
import sys
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick eval of Syscall Transformer")
    p.add_argument(
        "--aggregation",
        choices=["mean", "max", "p95"],
        default="mean",
        help="NLL aggregation per window (default: mean)",
    )
    p.add_argument(
        "--max-normal",
        type=int,
        default=200,
        help="Number of normal recordings to evaluate (default: 200). Use 0 for all.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32). Reduce if CUDA OOM.",
    )
    p.add_argument(
        "--auto-batch",
        action="store_true",
        help="Automatically halve batch size on CUDA OOM until it works.",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to specific checkpoint. If omitted, uses the latest in ckpt_path.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: cuda or cpu (default: cuda)",
    )
    return p.parse_args()


def find_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    normal_scores = y_score[y_true == 0]
    if len(normal_scores) == 0:
        return float(np.min(y_score) - 1.0)
    sorted_ns = np.sort(normal_scores)[::-1]
    idx = min(int(np.floor(target_fpr * len(sorted_ns))), len(sorted_ns) - 1)
    return float(sorted_ns[idx])


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> int:
    args = parse_args()

    # --- Load model ---
    if args.ckpt:
        best_ckpt = Path(args.ckpt)
    else:
        ckpt_dir = Path(ckpt_path)
        checkpoints = list(ckpt_dir.glob("*.ckpt"))
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {ckpt_dir}", file=sys.stderr)
            return 1
        best_ckpt = max(checkpoints, key=os.path.getctime)

    print(f"Loading: {best_ckpt.name}")
    print(f"Aggregation: {args.aggregation} | Max normal: {args.max_normal} | Batch: {args.batch_size} | Device: {args.device}")

    model = SyscallTransformer.load_from_checkpoint(str(best_ckpt))
    model.eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # --- Discover files ---
    test_dir = npy_dir / "test"
    window_files = sorted(list(test_dir.glob("*_windows.npy")))
    exploit_files = [f for f in window_files if "_exploit_" in f.name]
    normal_files = [f for f in window_files if "_normal_" in f.name]

    if args.max_normal > 0 and len(normal_files) > args.max_normal:
        normal_files = normal_files[:args.max_normal]

    target_files = sorted(exploit_files + normal_files)
    print(f"Evaluating {len(target_files)} recordings ({len(exploit_files)} exploit, {len(normal_files)} normal)...")

    # --- Inference ---
    all_window_scores = []
    all_window_labels = []
    recording_results = []

    current_batch_size = args.batch_size
    oom_occurred = False

    with torch.no_grad():
        pbar = tqdm(target_files, desc="Inference")
        for f_win in pbar:
            windows_np = np.load(f_win).astype(np.int64)
            f_label = f_win.parent / f_win.name.replace("_windows.npy", "_labels.npy")
            if not f_label.exists():
                continue
            window_gt_labels = np.load(f_label).astype(np.int8)

            num_windows = windows_np.shape[0]
            window_scores_list = []

            start_idx = 0
            while start_idx < num_windows:
                end_idx = min(start_idx + current_batch_size, num_windows)
                x_batch = torch.from_numpy(windows_np[start_idx:end_idx]).to(args.device)
                try:
                    scores_batch = model.compute_anomaly_score(x_batch, aggregation=args.aggregation)
                    window_scores_list.append(scores_batch.cpu().numpy())
                    start_idx = end_idx
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and args.auto_batch and current_batch_size > 1:
                        if not oom_occurred:
                            print(f"\n  CUDA OOM with batch={current_batch_size}, auto-reducing...")
                            oom_occurred = True
                        current_batch_size = max(1, current_batch_size // 2)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

            if oom_occurred:
                pbar.set_postfix({"batch": current_batch_size})

            window_scores = np.concatenate(window_scores_list)
            all_window_scores.extend(window_scores)
            all_window_labels.extend(window_gt_labels)

            is_exploit_rec = 1 if "_exploit_" in f_win.name else 0
            rec_score = window_scores.max()
            recording_results.append({"score": rec_score, "label": is_exploit_rec})

    # --- Metrics ---
    win_scores = np.array(all_window_scores)
    win_labels = np.array(all_window_labels)
    rec_scores = np.array([r["score"] for r in recording_results])
    rec_labels = np.array([r["label"] for r in recording_results])

    win_auroc = roc_auc_score(win_labels, win_scores)
    rec_auroc = roc_auc_score(rec_labels, rec_scores)

    # Best unconstrained F1
    prec, rec, thresh = precision_recall_curve(win_labels, win_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = np.argmax(f1)
    best_f1 = f1[best_idx]
    best_thr = thresh[best_idx] if best_idx < len(thresh) else 1.0

    # FPR-constrained thresholds
    fpr_targets = [0.01, 0.05, 0.10]
    win_fpr_results = [evaluate_at_threshold(win_labels, win_scores, find_threshold_at_fpr(win_labels, win_scores, t)) for t in fpr_targets]
    rec_fpr_results = [evaluate_at_threshold(rec_labels, rec_scores, find_threshold_at_fpr(rec_labels, rec_scores, t)) for t in fpr_targets]

    # --- Print ---
    print("\n" + "=" * 50)
    print("   QUICK EVAL RESULTS")
    print("=" * 50)
    print(f"Aggregation:           {args.aggregation}")
    print(f"Window-level AUROC:    {win_auroc:.4f}")
    print(f"Recording-level AUROC: {rec_auroc:.4f}")
    print(f"Best Window F1:        {best_f1:.4f} (thr={best_thr:.4f})")
    print("-" * 50)
    print("Window-level @ target FPR:")
    for r in win_fpr_results:
        print(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}")
    print("-" * 50)
    print("Recording-level @ target FPR:")
    for r in rec_fpr_results:
        print(f"  FPR<={r['fpr']:.4f} (thr={r['threshold']:.4f}): Recall={r['recall']:.4f}, Prec={r['precision']:.4f}, F1={r['f1']:.4f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
