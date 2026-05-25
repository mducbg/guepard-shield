"""P2 Assumption Check — Label Contamination at the Exploit Boundary.

The labeling rule: window is ATTACK if last-syscall timestamp >= exploit_start.
This marks the entire TAIL of a recording as attack, even though the actual
exploit may span only a brief burst of syscalls, after which behavior returns
to normal. Most labeled-attack windows contain normal-looking syscalls.

This script quantifies how severe the contamination is and finds the
"true detection window" (where the model score actually spikes) vs the
labeling boundary.

Run:
    uv run notebooks/p2/analyze_label_contamination.py

Outputs:
    results/p2/analysis/label_contamination/
        stats.json          — per-recording contamination statistics
        summary.txt         — printed summary
        score_timeline.png  — score timelines for representative recordings
        contamination_dist.png — distribution of contamination fractions
        detection_lag.png   — histogram of detection lag (jump_pos)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "p2"
SCORE_DIR = PROJECT_ROOT / "results" / "p2" / "scores"
OUT_DIR   = PROJECT_ROOT / "results" / "p2" / "analysis" / "label_contamination"

ORACLE_TAU = 2.606   # from WALKTHROUGH: oracle F1 threshold
WINDOW_SIZE = 64
STRIDE = 32


def load_data():
    scores  = np.load(SCORE_DIR / "test_last.npy",    mmap_mode="r")
    labels  = np.load(SCORE_DIR / "test_labels.npy",  mmap_mode="r")
    rec_ids = np.load(DATA_DIR  / "test_rec_ids.npy", mmap_mode="r")
    return scores, labels, rec_ids


def split_by_recording(rec_ids):
    starts = np.r_[0, np.flatnonzero(np.diff(rec_ids)) + 1]
    ends   = np.r_[starts[1:], len(rec_ids)]
    return list(zip(starts.tolist(), ends.tolist()))


def analyze_recording(seg_scores, seg_labels):
    """Return per-recording contamination stats, or None if not an attack rec."""
    if not seg_labels.any():
        return None

    exploit_idx = int(np.argmax(seg_labels))   # first labeled-attack window
    n_total   = len(seg_labels)
    n_normal  = exploit_idx
    n_labeled_atk = int(seg_labels.sum())

    # Windows labeled attack that score BELOW oracle threshold (look normal)
    atk_scores = seg_scores[exploit_idx:]
    n_undetected = int((atk_scores < ORACLE_TAU).sum())
    frac_undetected = n_undetected / max(n_labeled_atk, 1)

    # Detection lag: position of first window after exploit_start where score >= tau
    above = atk_scores >= ORACLE_TAU
    detection_lag = int(above.argmax()) if above.any() else -1

    # Score at boundary vs peak
    pre_window = seg_scores[max(0, exploit_idx - 10):exploit_idx]
    post_window = atk_scores[:min(50, len(atk_scores))]

    return {
        "n_total":         n_total,
        "n_normal":        n_normal,
        "exploit_start":   exploit_idx,
        "exploit_frac":    n_labeled_atk / n_total,
        "n_labeled_atk":   n_labeled_atk,
        "n_undetected":    n_undetected,
        "frac_undetected": frac_undetected,
        "detection_lag":   detection_lag,
        "score_pre_mean":  float(pre_window.mean()) if len(pre_window) else 0.0,
        "score_post_peak": float(post_window.max()) if len(post_window) else 0.0,
        "score_post_mean": float(post_window.mean()) if len(post_window) else 0.0,
    }


def plot_timelines(scores, labels, rec_bounds, stats, n_examples=12):
    """Plot score timelines for representative attack recordings."""
    # Pick: some with high contamination, some with low, some typical
    attack_indices = [i for i, s in enumerate(stats) if s is not None]
    fracs = np.array([stats[i]["frac_undetected"] for i in attack_indices])
    lags  = np.array([stats[i]["detection_lag"]    for i in attack_indices])

    # sample: low lag (fast detect), high lag (slow detect), lag=-1 (never detected)
    never_det   = [attack_indices[i] for i in range(len(attack_indices)) if lags[i] == -1][:3]
    fast_det    = [attack_indices[i] for i in np.where((lags >= 0) & (lags <= 2))[0]][:3]
    slow_det    = [attack_indices[i] for i in np.argsort(lags)[::-1] if lags[list(attack_indices).index(i) if i in attack_indices else 0] > 0][:3]
    typical_det = [attack_indices[i] for i in range(len(attack_indices))
                   if 5 <= lags[i] <= 30][:3]

    chosen = []
    for group, label in [(fast_det, "fast"), (slow_det, "slow"),
                          (typical_det, "typical"), (never_det, "never")]:
        for idx in group:
            if len(chosen) < n_examples:
                chosen.append((idx, label))

    if not chosen:
        return

    ncols = 3
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, (rec_idx, group_label) in enumerate(chosen):
        s, e = rec_bounds[rec_idx]
        seg_s = np.asarray(scores[s:e])
        seg_l = np.asarray(labels[s:e])
        exploit_idx = stats[rec_idx]["detection_lag"]
        label_start = stats[rec_idx]["exploit_start"]

        ax = axes[ax_idx]
        x = np.arange(len(seg_s))
        ax.plot(x, seg_s, color="steelblue", linewidth=0.6, alpha=0.8, label="NLL score")
        ax.axhline(ORACLE_TAU, color="red", linestyle="--", linewidth=1, label=f"τ={ORACLE_TAU}")
        ax.axvline(label_start, color="orange", linewidth=1.5, label="label boundary")
        if stats[rec_idx]["detection_lag"] >= 0:
            ax.axvline(label_start + stats[rec_idx]["detection_lag"],
                       color="green", linewidth=1.5, label="score spike")

        ax.set_title(
            f"rec={rec_idx} [{group_label}]  lag={stats[rec_idx]['detection_lag']}  "
            f"undetected={stats[rec_idx]['frac_undetected']*100:.0f}%",
            fontsize=8
        )
        ax.set_xlabel("window index", fontsize=7)
        ax.set_ylabel("NLL", fontsize=7)
        ax.legend(fontsize=6)
        ax.set_ylim(-0.2, min(seg_s.max() * 1.1, 10))

    for ax in axes[len(chosen):]:
        ax.set_visible(False)

    plt.suptitle(
        "Score timelines around exploit boundary\n"
        "Orange = label start, Green = first score > τ",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_timeline.png", dpi=150)
    plt.close()
    print(f"  saved → score_timeline.png")


def plot_contamination_dist(stats):
    fracs = np.array([s["frac_undetected"] for s in stats if s is not None])
    lags  = np.array([s["detection_lag"]   for s in stats if s is not None and s["detection_lag"] >= 0])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: contamination fraction distribution
    axes[0].hist(fracs, bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(np.median(fracs), color="red", linestyle="--",
                    label=f"median={np.median(fracs)*100:.1f}%")
    axes[0].set_xlabel("Fraction of labeled-attack windows with score < τ")
    axes[0].set_ylabel("Number of recordings")
    axes[0].set_title("Label contamination per recording\n(% attack-labeled windows that look normal)")
    axes[0].legend()

    # Panel 2: detection lag distribution
    axes[1].hist(lags, bins=np.arange(0, min(lags.max() + 2, 200), 1),
                 color="darkorange", edgecolor="white")
    axes[1].axvline(np.median(lags), color="red", linestyle="--",
                    label=f"median={np.median(lags):.0f} windows")
    axes[1].set_xlabel("Detection lag (windows after label boundary)")
    axes[1].set_ylabel("Number of recordings")
    axes[1].set_title(f"Detection lag (N={len(lags)} recs where score ever > τ)")
    axes[1].legend()
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "contamination_dist.png", dpi=150)
    plt.close()
    print(f"  saved → contamination_dist.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    scores, labels, rec_ids = load_data()
    rec_bounds = split_by_recording(np.asarray(rec_ids))
    n_normal_score = np.percentile(scores[labels == 0], 75)

    print(f"  Total windows : {len(scores):,}")
    print(f"  Attack windows: {labels.sum():,} ({labels.mean()*100:.1f}%)")
    print(f"  Recordings    : {len(rec_bounds):,}")

    print("\nAnalyzing per-recording contamination...")
    stats = []
    for s, e in rec_bounds:
        seg_s = np.asarray(scores[s:e])
        seg_l = np.asarray(labels[s:e])
        stats.append(analyze_recording(seg_s, seg_l))

    attack_stats = [s for s in stats if s is not None]
    fracs        = np.array([s["frac_undetected"] for s in attack_stats])
    lags         = np.array([s["detection_lag"]   for s in attack_stats])
    exploit_fracs = np.array([s["exploit_frac"]   for s in attack_stats])

    never_detected = int((lags == -1).sum())
    detectable     = int((lags >= 0).sum())
    lag_valid      = lags[lags >= 0]

    summary_lines = [
        "=" * 60,
        "LABEL CONTAMINATION ANALYSIS",
        "=" * 60,
        f"Oracle threshold τ = {ORACLE_TAU}  (oracle-optimal F1 from WALKTHROUGH)",
        "",
        f"Attack recordings   : {len(attack_stats):,}",
        "",
        "── What fraction of labeled-attack windows look NORMAL to the model? ──",
        f"  Median per recording : {np.median(fracs)*100:.1f}%",
        f"  Mean per recording   : {np.mean(fracs)*100:.1f}%",
        f"  p10 / p90            : {np.percentile(fracs,10)*100:.1f}% / {np.percentile(fracs,90)*100:.1f}%",
        "",
        "── Labeling structure ──",
        f"  Exploit fraction per recording (median): {np.median(exploit_fracs)*100:.1f}%",
        f"  → Labels mark the entire tail as attack, not just the exploit burst",
        "",
        "── Detectability ──",
        f"  Recordings where model NEVER detects (score always < τ): {never_detected} ({never_detected/len(attack_stats)*100:.1f}%)",
        f"  Recordings where model detects at least once           : {detectable} ({detectable/len(attack_stats)*100:.1f}%)",
        "",
        f"  Of detectable recordings:",
        f"    Median detection lag : {np.median(lag_valid):.0f} windows after label boundary",
        f"    Mean detection lag   : {np.mean(lag_valid):.0f} windows",
        f"    lag = 0 (immediate)  : {(lag_valid == 0).sum()} recordings",
        f"    lag ∈ [1,5]          : {((lag_valid >= 1) & (lag_valid <= 5)).sum()} recordings",
        f"    lag > 20             : {(lag_valid > 20).sum()} recordings",
        "",
        "── Implication for P3 DFA evaluation ──",
        f"  ~{np.median(fracs)*100:.0f}% of labeled-attack windows the DFA will CORRECTLY accept",
        f"  (they contain normal behavior; labels are wrong).",
        f"  Naively measured FPR and TPR will both be misleading.",
        f"  Recommendation: use a 'corrected' label that only marks windows",
        f"  where the score spike actually occurs.",
        "=" * 60,
    ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary + "\n")
    print(f"\n  saved → summary.txt")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(attack_stats[:500], f, indent=2)   # save first 500 for inspection
    print(f"  saved → stats.json (first 500 attack recordings)")

    print("\nGenerating plots...")
    plot_contamination_dist(stats)
    plot_timelines(scores, labels, rec_bounds, stats, n_examples=12)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
