# P2 Teacher Model Design

**Date:** 2026-04-09
**Dataset:** LID-DS-2021 only
**Goal:** Train a binary classifier (normal=0, attack=1) at recording level. Produces a teacher checkpoint used as the signal source for P3 rule extraction.

---

## Diagnostic Findings That Drive This Design

| Finding | Value | Implication |
|---------|-------|-------------|
| Median sequence length | 91,505 syscalls | Full-sequence Transformer infeasible — must window |
| Sequence p95 | 832,025 syscalls | Max window budget ~512–1024 |
| Bigram Jaccard (normal vs attack) | 0.811 (high) | Syscall-bigram GNN too weak — graphs nearly identical |
| Attack-only bigrams | 422 / 6,761 (6.2%) | Few unique transition patterns; attention > graph |
| Thread KL divergence mean | 1.48, 53.8% > 1.0 | Moderate per-thread separation — worth exploiting |
| Max threads per recording | up to 1,239 (ZipSlip) | Full thread-graph infeasible; clustering required |
| Attack timing mean fraction | 0.60 | Attacks in active phase; window labeling using timestamps is precise |
| Class imbalance | 6–10:1 normal:attack | Weighted BCE loss required |
| OOV syscalls | `pread`, `pwrite` only | Alias to `read`/`write` before encoding |

---

## Shared Design Decisions

**Framework:** Keras 3 with JAX backend (matches `pyproject.toml`; simpler training loop and built-in LSTM layers; `encode()` for P3 via sub-model pattern).

**Vocab:** `syscall_64.tbl` → 336 IDs + UNK=335. Pre-processing aliases `pread`→`read`, `pwrite`→`write` to eliminate the two dominant OOV names.

**Loss:** `BinaryCrossentropy` with `class_weight={0: 1.0, 1: n_normal/n_attack}` computed per split.

**Optimizer:** Adam, lr=1e-3 with cosine decay to 1e-5 over training.

**Early stopping:** `monitor="val_f1"`, `patience=5`, `mode="max"`.

**Checkpoint:** Save best val-F1 weights to `results/p2/approach_{a,b}/best.weights.h5`.

**Evaluation:** After each epoch — F1, AUROC, precision, recall globally and per-scenario (15 scenarios). Recording-level threshold tuned on val set via Youden's J statistic.

---

## Window Labeling (Both Approaches)

For attack recordings, the label is not applied to the full recording — only the portion at or after the exploit start:

- `exploit_start_time` comes from `metadata["time"]["exploit"][0]["absolute"]`
- A window covering syscall indices `[i : i+512]` gets **label=1** if `timestamps[i+511] >= exploit_start_time`
- All windows in normal recordings get label=0
- This gives precise per-window supervision rather than treating the entire attack recording as anomalous

---

## Approach A — Sliding-Window BiLSTM

### Architecture

```
Input: window of 512 syscall IDs  →  (batch, 512)
  Embedding(vocab=336, dim=64)    →  (batch, 512, 64)
  BiLSTM(hidden=256, layers=2, dropout=0.3)
  last hidden state (fwd+bwd concat)  →  (batch, 512)
  Linear(512 → 1)  →  sigmoid
```

The `encode()` sub-model returns the 512-dim pre-logit vector for P3.

**Recording-level inference:** max-pool window logits across all windows in the recording. Threshold tuned on val set.

### Data Pipeline

- Flatten recording syscall sequence (TIDs ignored)
- Slide window=512, stride=256 over the sequence
- Discard windows shorter than 512 (tail)
- Training: subsample 500K windows/epoch (stratified by label) to cap epoch time
- Approximate window count: median 91K syscalls / 256 stride ≈ 355 windows/recording × 10,518 train recordings ≈ 3.7M total (before subsampling)

### Hyperparameters

| Param | Value |
|-------|-------|
| Window size | 512 |
| Stride | 256 |
| Embedding dim | 64 |
| BiLSTM hidden | 256 |
| BiLSTM layers | 2 |
| Dropout | 0.3 |
| Batch size | 256 windows |
| Learning rate | 1e-3 → 1e-5 cosine |
| Max epochs | 50 |
| Early stop patience | 5 (val F1) |
| Windows/epoch (train) | 500K (stratified subsample) |

---

## Approach B — Hierarchical Per-Thread BiLSTM

### Architecture

```
Input: K thread sequences, each padded to 512 syscall IDs  →  (K, 512)

[Per-thread encoder — shared weights]
  Embedding(vocab=336, dim=64)        →  (K, 512, 64)
  BiLSTM(hidden=128, layers=1)
  last hidden state (fwd+bwd concat)  →  (K, 256)
  = thread_vecs

[Cross-thread attention pooling]
  query = learnable vector (256-dim)
  scores = softmax(query · thread_vecs^T / √256)   →  (K,)
  recording_vec = scores · thread_vecs              →  (256,)

[Classifier head]
  Linear(256 → 64)  →  ReLU  →  Linear(64 → 1)  →  sigmoid
```

The `encode()` sub-model returns the 256-dim `recording_vec` for P3.

**Recording-level inference:** direct sigmoid output. Threshold tuned on val set.

### Thread Handling

**If n_threads ≤ 64:** use threads directly, pad to K=64 with zero vectors.

**If n_threads > 64 (ZipSlip=1239, EPS_CWE-434=532, CVE-2020-13942=358, CVE-2012-2122=1025):**
1. Compute unigram frequency vector (336-dim, L1-normalised) per thread
2. k-means clustering with k=8 on the frequency vectors
3. For each cluster: concatenate member thread syscall sequences, truncate to 512
4. Feed k=8 cluster sequences through the per-thread BiLSTM
5. K=8 in the attention layer for these recordings

Clustering is precomputed once before training and cached to disk (`results/p2/approach_b/thread_clusters/`).

### Data Pipeline

- One sample = one full recording (K×512 syscall matrix)
- Label: recording-level (1 if `exploit_start_time` exists in metadata, else 0)
- No windowing; no subsampling — ~10,518 train recordings/epoch
- Batch size=32 (K×512 matrix per recording → ~32×64×512 = ~1M tokens/batch)

### Hyperparameters

| Param | Value |
|-------|-------|
| Max threads K | 64 |
| Cluster k (high-thread fallback) | 8 |
| Thread sequence length | 512 (truncate/pad) |
| Embedding dim | 64 |
| BiLSTM hidden | 128 |
| BiLSTM layers | 1 |
| Attention dim | 256 |
| Dropout | 0.3 |
| Batch size | 32 recordings |
| Learning rate | 1e-3 → 1e-5 cosine |
| Max epochs | 50 |
| Early stop patience | 5 (val F1) |

---

## File Layout

```
src/gp/
  models/
    bilstm.py          # SlidingWindowBiLSTM (approach A)
    hier_bilstm.py     # HierarchicalBiLSTM + ThreadClusterer (approach B)

notebooks/p2/
  1_train_a.py         # Train A: data pipeline, training loop, per-scenario eval, save checkpoint
  2_train_b.py         # Train B: clustering precompute, training loop, per-scenario eval, save checkpoint
  3_compare.py         # Side-by-side comparison table A vs B per scenario

results/p2/
  approach_a/
    best.weights.h5
    train_history.csv
    val_metrics_per_scenario.csv
  approach_b/
    best.weights.h5
    train_history.csv
    val_metrics_per_scenario.csv
    thread_clusters/   # cached cluster assignments per recording
```

---

## Evaluation Output

Both notebooks produce:

1. **`train_history.csv`** — loss, F1, AUROC per epoch (train + val)
2. **`val_metrics_per_scenario.csv`** — F1, AUROC, precision, recall per scenario on val set at best epoch
3. **`3_compare.py`** — prints a side-by-side table of all 15 scenarios + overall, A vs B

---

## What This Feeds Into

- **P3 rule extraction:** uses the `encode()` sub-model to produce embeddings; feeds SHAP or imodels on top of the teacher's latent space
- **Negative result to report:** bigram Jaccard=0.811 explicitly rules out GNN as the teacher architecture — documented in the diagnostic results
