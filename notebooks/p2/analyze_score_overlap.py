"""P2 Assumption Check — Score Distribution Overlap and Corrected Evaluation.

P3 needs the score (or accept/reject decision) to discriminate between normal
and attack windows. This script answers:

  1. How much do normal and attack score distributions overlap?
  2. What does the distribution look like if we remove contaminated labels
     (attack windows that the model correctly treats as normal)?
  3. What is the "true" AUROC if we only count windows where the exploit
     syscall signature is actually present (score-spike windows)?

The "corrected label" approach: a window gets label=1 only if it is within
the first SPIKE_WINDOW windows after the score crosses tau in an attack
recording. All other attack-labeled windows become label=0 for evaluation.

Run:
    uv run notebooks/p2/analyze_score_overlap.py

Outputs:
    results/p2/analysis/score_overlap/
        summary.txt
        score_distributions.png   — KDE of normal vs attack (raw + corrected)
        auroc_comparison.png      — AUROC: raw labels vs corrected labels
        score_by_lag.png          — mean score vs position after exploit boundary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "p2"
SCORE_DIR = PROJECT_ROOT / "results" / "p2" / "scores"
OUT_DIR   = PROJECT_ROOT / "results" / "p2" / "analysis" / "score_overlap"

ORACLE_TAU  = 2.606
SPIKE_WINDOW = 20   # windows around first score spike considered "true attack"


def load_data():
    scores  = np.load(SCORE_DIR / "test_last.npy",    mmap_mode="r")
    labels  = np.load(SCORE_DIR / "test_labels.npy",  mmap_mode="r")
    rec_ids = np.load(DATA_DIR  / "test_rec_ids.npy", mmap_mode="r")
    return scores, labels, rec_ids


def build_corrected_labels(scores, labels, rec_ids) -> np.ndarray:
    """Return corrected labels: attack only if within SPIKE_WINDOW of first score > tau.

    For recordings where the model never scores > tau (never detected),
    all attack-labeled windows get corrected label = 0.
    """
    corrected = np.asarray(labels).copy()
    starts = np.r_[0, np.flatnonzero(np.diff(rec_ids)) + 1]
    ends   = np.r_[starts[1:], len(rec_ids)]

    for s, e in zip(starts, ends):
        seg_l = np.asarray(labels[s:e])
        seg_s = np.asarray(scores[s:e])

        if not seg_l.any():
            continue

        exploit_idx = int(np.argmax(seg_l))
        atk_scores  = seg_s[exploit_idx:]
        above       = atk_scores >= ORACLE_TAU

        if not above.any():
            # Never detected: zero out all attack labels
            corrected[s + exploit_idx:e] = 0
            continue

        spike_pos = int(above.argmax())
        # Keep only SPIKE_WINDOW windows around the spike as true attack
        true_start = exploit_idx + max(0, spike_pos - 2)
        true_end   = exploit_idx + spike_pos + SPIKE_WINDOW
        # Zero out everything outside [true_start, true_end]
        corrected[s + exploit_idx:s + true_start] = 0
        corrected[s + true_end:e] = 0

    return corrected


def plot_score_distributions(scores, labels, corrected_labels):
    norm_scores  = np.asarray(scores[labels == 0])
    atk_scores   = np.asarray(scores[labels == 1])
    catk_scores  = np.asarray(scores[corrected_labels == 1])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Raw labels ────────────────────────────────────────────────────────────
    ax = axes[0]
    # Clip for readability
    clip = 8.0
    bins = np.linspace(0, clip, 200)
    ax.hist(np.clip(norm_scores, 0, clip), bins=bins, density=True,
            alpha=0.6, color="steelblue", label=f"Normal ({len(norm_scores):,})")
    ax.hist(np.clip(atk_scores,  0, clip), bins=bins, density=True,
            alpha=0.6, color="crimson",   label=f"Attack (raw, {len(atk_scores):,})")
    ax.axvline(ORACLE_TAU, color="black", linestyle="--", linewidth=1.5, label=f"τ={ORACLE_TAU}")
    ax.set_title("Raw labels\n(17.2% of test windows labeled attack)")
    ax.set_xlabel("Last-token NLL")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, clip)

    # ── Corrected labels ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.hist(np.clip(norm_scores,  0, clip), bins=bins, density=True,
            alpha=0.6, color="steelblue", label=f"Normal ({len(norm_scores):,})")
    ax.hist(np.clip(catk_scores,  0, clip), bins=bins, density=True,
            alpha=0.6, color="darkorange",
            label=f"Attack (corrected, {len(catk_scores):,})")
    ax.axvline(ORACLE_TAU, color="black", linestyle="--", linewidth=1.5, label=f"τ={ORACLE_TAU}")
    ax.set_title(f"Corrected labels\n(only ±{SPIKE_WINDOW} windows around score spike)")
    ax.set_xlabel("Last-token NLL")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, clip)

    plt.suptitle("Score distribution: normal vs attack", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_distributions.png", dpi=150)
    plt.close()
    print("  saved → score_distributions.png")


def plot_roc_comparison(scores, labels, corrected_labels):
    scores_arr = np.asarray(scores)

    fig, ax = plt.subplots(figsize=(7, 6))

    for lbl_arr, color, name in [
        (labels,            "steelblue",  "Raw labels"),
        (corrected_labels,  "darkorange", f"Corrected labels (±{SPIKE_WINDOW} spike windows)"),
    ]:
        lbl_arr = np.asarray(lbl_arr)
        if lbl_arr.sum() == 0:
            continue
        auc = roc_auc_score(lbl_arr, scores_arr)
        fpr, tpr, _ = roc_curve(lbl_arr, scores_arr)
        ax.plot(fpr, tpr, color=color, label=f"{name}  AUROC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve: raw vs corrected labels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "auroc_comparison.png", dpi=150)
    plt.close()
    print("  saved → auroc_comparison.png")


def plot_score_by_lag(scores, labels, rec_ids):
    """Mean score as a function of position RELATIVE to exploit boundary."""
    starts = np.r_[0, np.flatnonzero(np.diff(rec_ids)) + 1]
    ends   = np.r_[starts[1:], len(rec_ids)]

    MAX_LAG = 200
    buckets = np.zeros(MAX_LAG, dtype=np.float64)
    counts  = np.zeros(MAX_LAG, dtype=np.int64)

    for s, e in zip(starts, ends):
        seg_l = np.asarray(labels[s:e])
        seg_s = np.asarray(scores[s:e])
        if not seg_l.any():
            continue
        exploit_idx = int(np.argmax(seg_l))
        for offset in range(min(MAX_LAG, e - s - exploit_idx)):
            buckets[offset] += seg_s[exploit_idx + offset]
            counts[offset]  += 1

    valid = counts > 100
    mean_score = np.where(valid, buckets / np.maximum(counts, 1), np.nan)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(MAX_LAG)
    ax.plot(x[valid], mean_score[valid], color="crimson", linewidth=1.5)
    ax.axhline(ORACLE_TAU, color="black", linestyle="--", linewidth=1,
               label=f"τ={ORACLE_TAU}")
    ax.fill_between(x[valid], 0, mean_score[valid],
                    where=mean_score[valid] >= ORACLE_TAU,
                    alpha=0.3, color="crimson", label="detectable region")
    ax.set_xlabel("Window position after exploit label boundary")
    ax.set_ylabel("Mean NLL score")
    ax.set_title("Mean score vs. position after exploit_start\n"
                 "(averaged across all attack recordings)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_by_lag.png", dpi=150)
    plt.close()
    print("  saved → score_by_lag.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    scores, labels, rec_ids = load_data()
    scores_arr  = np.asarray(scores)
    labels_arr  = np.asarray(labels)
    rec_ids_arr = np.asarray(rec_ids)

    print("Building corrected labels...")
    corrected = build_corrected_labels(scores_arr, labels_arr, rec_ids_arr)

    norm_scores  = scores_arr[labels_arr == 0]
    atk_raw      = scores_arr[labels_arr == 1]
    atk_corr     = scores_arr[corrected  == 1]

    auroc_raw  = roc_auc_score(labels_arr, scores_arr)
    # Ghi chú: KHÔNG tính AUROC trên corrected labels vì đó là circular reasoning
    # (corrected labels được định nghĩa dựa trên score của model).

    # Overlap stats
    p50_normal = np.median(norm_scores)
    p99_normal = np.percentile(norm_scores, 99)
    overlap_at_p50 = (atk_raw < p50_normal).mean()
    overlap_at_p99 = (atk_raw < p99_normal).mean()
    overlap_at_tau = (atk_raw < ORACLE_TAU).mean()

    corr_overlap_at_tau = (atk_corr < ORACLE_TAU).mean() if len(atk_corr) > 0 else float("nan")

    lines = [
        "=" * 60,
        "SCORE DISTRIBUTION OVERLAP ANALYSIS",
        "=" * 60,
        f"τ (oracle) = {ORACLE_TAU}",
        "",
        "── Raw label statistics ──",
        f"  Normal windows      : {len(norm_scores):,}",
        f"  Attack windows (raw): {len(atk_raw):,} ({labels_arr.mean()*100:.1f}%)",
        f"  AUROC (raw labels)  : {auroc_raw:.4f}  ← con số hợp lệ duy nhất",
        "",
        "  Overlap (attack windows with score BELOW normal threshold):",
        f"    < median_normal ({p50_normal:.3f}) : {overlap_at_p50*100:.1f}%",
        f"    < p99_normal    ({p99_normal:.3f}) : {overlap_at_p99*100:.1f}%",
        f"    < oracle τ      ({ORACLE_TAU:.3f}) : {overlap_at_tau*100:.1f}%",
        "",
        f"── Corrected label statistics (±{SPIKE_WINDOW} windows around spike) ──",
        f"  Attack windows (corrected): {int(corrected.sum()):,} ({corrected.mean()*100:.2f}%)",
        f"  Overlap at τ (corrected)  : {corr_overlap_at_tau*100:.1f}%",
        "",
        "── Lưu ý quan trọng về AUROC ──",
        f"  KHÔNG tính 'AUROC corrected' vì corrected labels được định nghĩa",
        f"  dựa trên score của chính model => circular reasoning.",
        f"  AUROC={auroc_raw:.4f} trên raw labels là ground truth duy nhất.",
        f"  Diễn giải: {overlap_at_tau*100:.0f}% labeled-attack windows có score thấp",
        f"  (hành vi bình thường post-exploit), kéo AUROC xuống so với",
        f"  khả năng thực của model trên các windows thực sự anomalous.",
        "",
        "  For P3 DFA evaluation: dùng per-recording detection rate,",
        "  không dùng per-window accuracy trên raw labels.",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print("  saved → summary.txt")

    print("\nGenerating plots...")
    plot_score_distributions(scores_arr, labels_arr, corrected)
    plot_roc_comparison(scores_arr, labels_arr, corrected)
    plot_score_by_lag(scores_arr, labels_arr, rec_ids_arr)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
