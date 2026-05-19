"""Anomaly evaluation metrics for Phase 2.

Anomaly score for window [s_1, …, s_W]:
    score = -log P(s_W | s_1, …, s_{W-1})   (last-token NLL)

Inference is handled by SyscallTransformer.predict_step + Lightning Trainer.predict().
This module only contains threshold selection and metric computation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)


def select_threshold(val_scores: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    """Return (percentile_threshold, mu+3sigma_threshold)."""
    tau_pct = float(np.percentile(val_scores, percentile))
    tau_sig = float(val_scores.mean() + 3 * val_scores.std())
    return tau_pct, tau_sig


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute AUROC, PR-AUC, FPR, and per-class classification report."""
    preds = (scores >= threshold).astype(int)

    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())

    auroc  = float(roc_auc_score(labels, scores))           if n_pos > 0 and n_neg > 0 else float("nan")
    pr_auc = float(average_precision_score(labels, scores)) if n_pos > 0               else float("nan")

    fp  = int(((preds == 1) & (labels == 0)).sum())
    fpr = fp / n_neg if n_neg > 0 else float("nan")

    report = classification_report(labels, preds, target_names=["normal", "attack"], zero_division=0)

    return {"auroc": auroc, "pr_auc": pr_auc, "fpr": fpr, "_report": report}
