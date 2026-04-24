# Tier 2 Design Doc — One-Class Sequence Model

**Context:** If Tier 1 improvements (more training data, max/p95 aggregation, finer eval stride) do not push window-level AUROC above ~0.90, we switch to a stronger one-class architecture.

**Goal:** Learn a compact normality representation from LID-DS-2021 normal-only training data, producing a scalar anomaly score with a cleaner decision boundary than next-token perplexity.

---

## 1. Why Next-Token Perplexity Hits a Ceiling

- The causal Transformer LM learns **local token-transition probabilities**.
- Many attacks in LID-DS-2021 use valid syscalls in locally-valid order (e.g., `open → read → write → close`).
- Mean-NLL over 1,000 tokens dilutes short attack segments.
- Result: ~24% of attack windows score within the normal range, creating an **unavoidable overlap** for a pure LM.

A **global sequence representation** (entire window → single vector → reconstruction error / distance to center) can capture **structural deviations** that are invisible at the token level.

---

## 2. Proposed Architecture: Transformer Autoencoder + Deep SVDD

### 2.1 Model

```
Input: [batch, seq_len] syscall IDs
  ↓
Embedding + Positional Encoding
  ↓
Transformer Encoder (4 layers, 256 d_model)   →  [batch, seq_len, d_model]
  ↓
Mean-pool over seq_len (or [CLS] token)       →  [batch, d_model]  (latent z)
  ↓
Transformer Decoder (4 layers, 256 d_model)   →  [batch, seq_len, d_model]
  ↓
Linear → vocab logits                         →  reconstruction
```

**Loss:**

```python
# Reconstruction loss (cross-entropy)
L_recon = CrossEntropyLoss(reconstructed_logits, input_tokens)

# One-class center loss (Deep SVDD)
# Pre-compute center c on a forward pass over training data
L_svdd = mean( ||z_i - c||^2 )

# Total
L = L_recon + λ * L_svdd
```

- `λ` starts at 0 (pure autoencoder for warm-up), ramps to 1e-3 after 5 epochs.
- `c` is fixed after epoch 0 (computed on a random subset of train data with gradient off).

### 2.2 Anomaly Score

```python
score = ||z - c||^2    # Euclidean distance to normality center
```

This is a **single scalar per window** with a clear geometric interpretation: the farther from the center, the more anomalous.

### 2.3 Why This Beats the LM for HIDS

| Aspect | Causal LM | AE + SVDD |
|---|---|---|
| What it models | P(token_t \| token_<t) | Global structure of normal windows |
| Attack sensitivity | Local token surprise | Global sequence deviation |
| Score range | [0, ∞] unbounded (NLL) | [0, ∞] but naturally peaked at center |
| Thresholding | Hard (overlapping long tails) | Easier (normal cluster is compact) |
| Rule distillation | Noisy teacher labels | Cleaner pseudo-labels |

---

## 3. Training Protocol

1. **Pre-train** the autoencoder on LID-DS-2021 train/val (normal only) for 20 epochs with reconstruction loss only.
2. **Compute center** `c` by forward-passing 10K random train windows and taking the mean of their latent vectors.
3. **Fine-tune** 10 more epochs with `L_recon + λ * L_svdd`.
4. **Validate** on LID-DS-2021 val (normal only): monitor reconstruction loss; early-stop if it rises.
5. **Evaluate** on LID-DS-2021 test with the same protocol as Tier 1.

**Hyperparameters (minimal changes from Tier 1):**
- `d_model=256`, `num_layers=4`, `nhead=8`
- `window_size=1000`, `stride_train=1000`, `stride_eval=200`
- `batch_size=64`, `lr=1e-3`, `λ=1e-3`
- `max_windows_train=None` (use all)

---

## 4. Data Pipeline Changes

- Re-use `SyscallDataModule` but remove the next-token shifting logic.
- Input = target = full window (no `[:, :-1]` / `[:, 1:]` split).
- Add a `compute_latent(batch)` method to extract `z` for center computation and scoring.

---

## 5. Expected Outcomes

| Metric | Tier 1 (best case) | Tier 2 (target) |
|---|---|---|
| Window AUROC | 0.90–0.92 | **> 0.95** |
| Recording AUROC | ~0.98 | **> 0.99** |
| FPR@Recall=0.90 | ~5% | **< 1%** |

**Risk:** If normal windows already form a highly multi-modal distribution (many different legitimate server behaviors), a single center `c` may be too simplistic. Mitigation: use a small set of K-means centers instead of one SVDD center, or use a VAE with a probabilistic latent space.

---

## 6. Implementation Checklist (if Tier 1 fails)

- [ ] Implement `SyscallAutoencoder` (encoder + decoder + latent pool)
- [ ] Add `compute_latent()` and `compute_anomaly_score()` methods
- [ ] Add center-computation hook in `on_train_epoch_end`
- [ ] Write `train_autoencoder.py` (replaces `train_transformer.py`)
- [ ] Write `evaluate_autoencoder.py` (same eval protocol)
- [ ] Run ablation: reconstruction-only vs reconstruction+SVDD
- [ ] If still < 0.95 AUROC, try K-means centers or VAE

**Estimated effort:** 3–5 days (mostly model rewrite; data pipeline stays the same).

---

## 7. Decision Gate

> **Trigger Tier 2 if and only if:**
> After re-training with Tier 1 changes (`max_windows_train=None`, `aggregation=max`, `stride_eval=200`), window-level AUROC on LID-DS-2021 test remains **< 0.90**.

If Tier 1 reaches **≥ 0.90**, skip Tier 2 and proceed directly to Phase 3 (Rule Extraction) with the improved Transformer as Teacher.
