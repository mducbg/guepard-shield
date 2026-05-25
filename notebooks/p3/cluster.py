"""Phase 3 — Step 2: K-Means clustering of hidden states.

Run from project root:
    uv run notebooks/p3/cluster.py

For each K in {64, 128, 256, 512}, fits MiniBatchKMeans on train_H.dat
and saves cluster labels and centroids.

Outputs (results/p3/clusters/K{k}/):
    labels.dat      [M]       int32 memmap — cluster ID per window
    centroids.npy   [K, 128]  float32      — cluster centres
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

HIDDEN_DIR = PROJECT_ROOT / "results" / "p3" / "hidden_states"
CLUSTER_DIR = PROJECT_ROOT / "results" / "p3" / "clusters"

K_VALUES = [64, 128, 256, 512]
FIT_CHUNK  = 50_000   # rows per partial_fit call
PRED_CHUNK = 100_000  # rows per predict call


def load_info() -> dict:
    with open(HIDDEN_DIR / "info.json") as f:
        return json.load(f)


def run_k(K: int, H: np.memmap, M: int, out_dir: Path, H_dtype: str = "float32") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.dat"
    centroids_path = out_dir / "centroids.npy"

    print(f"\n── K={K} ────────────────────────────────────────")

    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=10_000,
        random_state=42,
        n_init=3,
        max_iter=100,
        verbose=0,
    )

    # Fit — upcast to float32 if H is stored as float16
    needs_cast = H_dtype != "float32"
    n_epochs = 3
    for epoch in range(n_epochs):
        for start in tqdm(
            range(0, M, FIT_CHUNK),
            desc=f"  fit epoch {epoch+1}/{n_epochs}",
            unit="chunk",
        ):
            end = min(start + FIT_CHUNK, M)
            chunk = np.asarray(H[start:end], dtype=np.float32) if needs_cast else H[start:end]
            km.partial_fit(chunk)

    # Predict labels
    labels_mmap = np.memmap(labels_path, dtype="int32", mode="w+", shape=(M,))
    for start in tqdm(range(0, M, PRED_CHUNK), desc="  predict", unit="chunk"):
        end = min(start + PRED_CHUNK, M)
        chunk = np.asarray(H[start:end], dtype=np.float32) if needs_cast else H[start:end]
        labels_mmap[start:end] = km.predict(chunk).astype(np.int32)
    del labels_mmap

    np.save(centroids_path, km.cluster_centers_.astype(np.float32))

    print(f"  saved labels → {labels_path}")
    print(f"  saved centroids → {centroids_path}")


def plot_cluster_distributions(M: int) -> None:
    plots_dir = PROJECT_ROOT / "results" / "p3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    for K in K_VALUES:
        labels_path = CLUSTER_DIR / f"K{K}" / "labels.dat"
        if not labels_path.exists():
            continue
        labels = np.memmap(labels_path, dtype="int32", mode="r", shape=(M,))
        counts = np.bincount(np.asarray(labels), minlength=K)
        # Sort counts descending
        counts_sorted = np.sort(counts)[::-1]
        # Normalize to percentage
        pct = (counts_sorted / M) * 100
        
        plt.plot(np.arange(1, K + 1), pct, label=f"K = {K}", marker='o', markersize=3)
        
    plt.xlabel("Cluster Rank (sorted by size)")
    plt.ylabel("Percentage of training windows (%)")
    plt.title("Cluster Size Distribution for different K values")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plot_path = plots_dir / "cluster_size_distributions.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved cluster size distribution plot → {plot_path}")


def main() -> None:
    info = load_info()
    M = info["M"]
    D = info["D_model"]
    H_path = HIDDEN_DIR / "train_H.dat"

    H_dtype = info.get("H_dtype", "float32")
    print(f"Loading hidden states from {H_path}  (M={M:,}, D={D}, dtype={H_dtype})")
    H = np.memmap(H_path, dtype=H_dtype, mode="r", shape=(M, D))

    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    for K in K_VALUES:
        out_dir = CLUSTER_DIR / f"K{K}"
        run_k(K, H, M, out_dir, H_dtype=H_dtype)

    print("\nAll K-Means runs complete.")
    try:
        plot_cluster_distributions(M)
    except Exception as e:
        print(f"Error plotting cluster distributions: {e}")


if __name__ == "__main__":
    main()
