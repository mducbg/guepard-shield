"""P2 Assumption Check — Hidden State Separability (Normal vs Attack).

Core P3 assumption: the Transformer's final-layer hidden states for normal
windows cluster separately from those for attack windows. K-Means over
these states only makes sense if this assumption holds.

This script:
  1. Samples N windows from test set (normal and attack balanced)
  2. Extracts hidden states via model.encode()
  3. Computes PCA and visualizes the 2D projection
  4. Measures a quantitative separability score (silhouette, centroid distance)

Note: only uses windows with score > oracle_tau as "true attack" to avoid
label contamination (see analyze_label_contamination.py).

Run:
    uv run notebooks/p2/analyze_hidden_sep.py --ckpt results/p2/checkpoints/best.ckpt

Outputs:
    results/p2/analysis/hidden_sep/
        summary.txt
        pca_normal_vs_attack.png     — PCA scatter plot
        pca_score_colored.png        — PCA colored by NLL score
        centroid_distances.png       — distance from normal centroid per group
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from gp.model import SyscallTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "p2"
SCORE_DIR = PROJECT_ROOT / "results" / "p2" / "scores"
OUT_DIR   = PROJECT_ROOT / "results" / "p2" / "analysis" / "hidden_sep"

ORACLE_TAU   = 2.606
N_SAMPLE     = 5_000   # per group (normal / attack_raw / attack_true)
BATCH_SIZE   = 512
RANDOM_SEED  = 42


def load_scores_and_ids():
    scores  = np.load(SCORE_DIR / "test_last.npy",    mmap_mode="r")
    labels  = np.load(SCORE_DIR / "test_labels.npy",  mmap_mode="r")
    rec_ids = np.load(DATA_DIR  / "test_rec_ids.npy", mmap_mode="r")
    return scores, labels, rec_ids


def sample_indices(scores, labels, n: int, rng) -> dict[str, np.ndarray]:
    """Return sampled indices for three groups."""
    normal_idx   = np.where(labels == 0)[0]
    atk_raw_idx  = np.where(labels == 1)[0]
    atk_true_idx = np.where((labels == 1) & (scores >= ORACLE_TAU))[0]

    def _sample(idx):
        if len(idx) <= n:
            return idx
        return rng.choice(idx, size=n, replace=False)

    return {
        "normal":     _sample(normal_idx),
        "atk_raw":    _sample(atk_raw_idx),
        "atk_true":   _sample(atk_true_idx),
    }


def encode_windows(
    model: SyscallTransformer,
    X: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Extract hidden states for a set of windows given their indices into X."""
    all_h = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        batch = torch.from_numpy(X[batch_idx].astype(np.int64)).to(device)
        with torch.inference_mode():
            h = model.encode(batch)
        all_h.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(all_h, axis=0)


def plot_pca_scatter(pcs, groups_pcs, group_labels, group_scores):
    """PCA 2D scatter: one panel colored by group, one by score."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"normal": "steelblue", "atk_raw": "orange", "atk_true": "crimson"}
    labels_map = {
        "normal":   "Normal",
        "atk_raw":  "Attack (labeled, incl. contaminated)",
        "atk_true": f"Attack (score ≥ τ={ORACLE_TAU})",
    }

    # Panel 1: colored by group
    ax = axes[0]
    for grp, pcs_g in groups_pcs.items():
        ax.scatter(pcs_g[:, 0], pcs_g[:, 1],
                   c=colors[grp], label=labels_map[grp],
                   s=4, alpha=0.4, rasterized=True)
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA: normal vs attack")
    ax.legend(markerscale=3)

    # Panel 2: colored by NLL score (all groups combined)
    ax = axes[1]
    all_pcs   = np.concatenate(list(groups_pcs.values()), axis=0)
    all_scores = np.concatenate(list(group_scores.values()), axis=0)
    sc = ax.scatter(all_pcs[:, 0], all_pcs[:, 1],
                    c=np.clip(all_scores, 0, 5),
                    cmap="RdYlBu_r", s=4, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="NLL score (clipped at 5)")
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA: colored by NLL score")

    plt.suptitle("Hidden state PCA — do normal and attack windows separate?", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_normal_vs_attack.png", dpi=150)
    plt.close()
    print("  saved → pca_normal_vs_attack.png")


def plot_centroid_distances(groups_h):
    """Distance from each group's hidden states to the normal centroid."""
    normal_centroid = groups_h["normal"].mean(axis=0)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"normal": "steelblue", "atk_raw": "orange", "atk_true": "crimson"}
    labels_map = {"normal": "Normal", "atk_raw": "Attack (raw)", "atk_true": "Attack (true)"}
    clip = 20.0
    bins = np.linspace(0, clip, 80)

    for grp, h in groups_h.items():
        dists = np.linalg.norm(h - normal_centroid, axis=1)
        ax.hist(np.clip(dists, 0, clip), bins=bins,
                color=colors[grp], label=f"{labels_map[grp]} (n={len(h):,})",
                alpha=0.55, density=True)

    ax.set_xlabel("L2 distance from normal centroid")
    ax.set_ylabel("Density")
    ax.set_title("Distance of hidden states from normal cluster centroid")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "centroid_distances.png", dpi=150)
    plt.close()
    print("  saved → centroid_distances.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best.ckpt")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading scores + labels...")
    scores, labels, rec_ids = load_scores_and_ids()
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    print(f"  Normal windows     : {(labels_arr==0).sum():,}")
    print(f"  Attack windows     : {(labels_arr==1).sum():,}")
    print(f"  True attack (≥ τ)  : {((labels_arr==1)&(scores_arr>=ORACLE_TAU)).sum():,}")

    indices = sample_indices(scores_arr, labels_arr, N_SAMPLE, rng)
    for grp, idx in indices.items():
        print(f"  Sampled {grp}: {len(idx):,}")

    print(f"\nLoading checkpoint from {args.ckpt}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: SyscallTransformer = SyscallTransformer.load_from_checkpoint(
        args.ckpt, map_location=device
    )
    model.eval()
    print(f"  Model on {device}")

    print("\nLoading test windows (mmap)...")
    X = np.load(DATA_DIR / "test_X.npy", mmap_mode="r")
    print(f"  test_X shape: {X.shape}")

    print("\nExtracting hidden states for each group...")
    groups_h: dict[str, np.ndarray] = {}
    groups_scores: dict[str, np.ndarray] = {}
    for grp, idx in indices.items():
        print(f"  Encoding {grp} ({len(idx)} windows)...")
        groups_h[grp]      = encode_windows(model, X, idx, device)
        groups_scores[grp] = scores_arr[idx]
        print(f"    hidden shape: {groups_h[grp].shape}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("\nFitting PCA on all groups combined...")
    all_h = np.concatenate(list(groups_h.values()), axis=0)
    scaler = StandardScaler()
    all_h_scaled = scaler.fit_transform(all_h)

    pca = PCA(n_components=10, random_state=RANDOM_SEED)
    pca.fit(all_h_scaled)

    groups_pcs = {}
    offset = 0
    for grp, h in groups_h.items():
        n = len(h)
        groups_pcs[grp] = pca.transform(all_h_scaled[offset:offset + n])[:, :2]
        offset += n

    print(f"  Explained variance (PC1+PC2): {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")

    # ── Separability metrics ──────────────────────────────────────────────────
    n_norm = len(groups_h["normal"])
    n_atk  = len(groups_h["atk_true"])

    normal_centroid = groups_h["normal"].mean(axis=0)
    atk_centroid    = groups_h["atk_true"].mean(axis=0)
    centroid_dist   = float(np.linalg.norm(normal_centroid - atk_centroid))

    # Silhouette on a subset (expensive on large sets)
    sil_X = np.concatenate([groups_h["normal"][:1000], groups_h["atk_true"][:1000]])
    sil_y = np.array([0] * 1000 + [1] * 1000)
    silhouette = silhouette_score(sil_X, sil_y, sample_size=2000, random_state=RANDOM_SEED)

    # Intra-group spread
    norm_spread = np.linalg.norm(groups_h["normal"] - normal_centroid, axis=1).mean()
    atk_spread  = np.linalg.norm(groups_h["atk_true"] - atk_centroid, axis=1).mean()

    # Fraction of attack-true hidden states closer to attack centroid than normal centroid
    d_to_norm = np.linalg.norm(groups_h["atk_true"] - normal_centroid, axis=1)
    d_to_atk  = np.linalg.norm(groups_h["atk_true"] - atk_centroid, axis=1)
    frac_closer_to_atk = (d_to_atk < d_to_norm).mean()

    lines = [
        "=" * 60,
        "HIDDEN STATE SEPARABILITY ANALYSIS",
        "=" * 60,
        f"  Normal windows sampled    : {n_norm:,}",
        f"  True-attack sampled       : {n_atk:,}",
        f"  PCA explained var (2D)    : {pca.explained_variance_ratio_[:2].sum()*100:.1f}%",
        "",
        "── Separability metrics ──",
        f"  Centroid distance (normal vs true-attack) : {centroid_dist:.4f}",
        f"  Normal intra-group spread (mean dist)     : {norm_spread:.4f}",
        f"  Attack intra-group spread (mean dist)     : {atk_spread:.4f}",
        f"  Silhouette score (1000+1000 sample)       : {silhouette:.4f}",
        f"    [-1=fully overlapping, 0=boundary, 1=perfectly separated]",
        "",
        f"  % true-attack hidden states closer to attack centroid: {frac_closer_to_atk*100:.1f}%",
        "",
        "── Interpretation for P3 ──",
    ]

    if silhouette > 0.5:
        lines.append("  GOOD: hidden states are well-separated. K-Means should produce")
        lines.append("  meaningful DFA states that distinguish normal from attack.")
    elif silhouette > 0.1:
        lines.append("  MARGINAL: some separation exists but overlap is significant.")
        lines.append("  K-Means states will capture partial behavioral differences.")
        lines.append("  Expect higher non-determinism in DFA transitions.")
    else:
        lines.append("  POOR: hidden states heavily overlap. K-Means clustering will")
        lines.append("  not produce states that correlate with attack/normal labels.")
        lines.append("  The DFA fidelity to the Transformer will be low.")

    lines.append("=" * 60)
    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print("  saved → summary.txt")

    print("\nGenerating plots...")
    plot_pca_scatter(pca, groups_pcs, None, groups_scores)
    plot_centroid_distances(groups_h)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
