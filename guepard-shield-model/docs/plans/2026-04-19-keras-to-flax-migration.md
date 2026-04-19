# Keras → Flax NNX Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the Teacher Transformer from Keras 3 to Flax NNX + JAX + Optax + Orbax, gaining native gradient accumulation, explicit training loop control, and direct Orbax checkpointing.

**Architecture:** Rewrite `SyscallTransformer` as `nnx.Module` with `nnx.MultiHeadAttention`, `nnx.Linear`, `nnx.LayerNorm`, `nnx.Embed`, `nnx.Dropout`. Custom training loop with `jax.jit` + `optax.MultiSteps` for gradient accumulation. Orbax `StandardCheckpointer` for save/load. Data stays as numpy mmap arrays — JAX handles host-to-device transfer.

**Tech Stack:** flax (nnx API), jax[cuda12], optax, orbax-checkpoint, numpy, polars, scikit-learn, rich

---

## Key Design Decisions

1. **Flax NNX (not Linen)** — NNX uses Python reference semantics, no `init`/`apply` split, no `variables.py` boilerplate. Eager init = simpler debugging.
2. **`nnx.jit` pattern** — use `@nnx.jit` with mutable model state for train step. `nnx.split/merge` pattern for eval step (functional, no side effects).
3. **Gradient accumulation** — `optax.MultiSteps` wraps the base optimizer. Every `grad_accum_steps` micro-batches, one real optimizer step fires. No custom callback hackery.
4. **Mixed precision** — JAX handles `bfloat16` matmul precision via `jax.default_matmul_precision("bfloat16")`. The output head remains `float32` by explicit dtype annotation on the `nnx.Linear`.
5. **Loss with PAD masking** — `optax.softmax_cross_entropy_with_integer_labels` + manual mask (multiply by `(y != 0)` mask then divide by valid count). No magic `ignore_class=0`.
6. **Data pipeline** — numpy mmap arrays, sliced on-host, transferred to device per batch in the training loop. No tf.data, no Grain.
7. **Checkpoint format** — Orbax `StandardCheckpointer` with `nnx.split(model)` → save state; load → `nnx.merge(graphdef, state)`. Compatible with future distributed training.
8. **Eval callback** — Keras `RecordingNTPCallback` → plain function `eval_epoch()` called in training loop. Simple.

---

## Migration Map

| Component | Keras 3 | Flax NNX |
|---|---|---|
| Model def | `keras.Model` subclass | `nnx.Module` subclass |
| Embedding | `keras.layers.Embedding` | `nnx.Embed` |
| Attention | `keras.layers.MultiHeadAttention` + causal mask | `nnx.MultiHeadAttention(is_causal=True)` |
| LayerNorm | `keras.layers.LayerNormalization(epsilon=1e-5)` | `nnx.LayerNorm(epsilon=1e-5)` |
| FFN | `keras.layers.Dense` | `nnx.Linear` |
| Dropout | `keras.layers.Dropout` | `nnx.Dropout` |
| Optimizer | `keras.optimizers.AdamW` | `optax.adamw` |
| LR schedule | `keras.optimizers.schedules.CosineDecay` | `optax.cosine_decay_schedule` |
| Grad accum | No-op (Keras broken) | `optax.MultiSteps(every_k_schedule=8)` |
| Loss | `SparseCategoricalCrossentropy(ignore_class=0)` | Manual softmax-xent + PAD mask |
| Checkpoint | `keras.callbacks.OrbaxCheckpoint` | `orbax.checkpoint.StandardCheckpointer` |
| Mixed precision | `keras.mixed_precision.set_global_policy` | `jax.default_matmul_precision("bfloat16")` |
| Training loop | `model.fit()` | Custom `for epoch... for batch...` |
| Eval callback | `RecordingNTPCallback` | `eval_epoch()` function in loop |

---

## Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Remove `keras`, add `flax`, `optax`. Keep everything else.

```toml
[project]
name = "guepard-shield-model"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "imodels>=2.0.4",
    "ipykernel>=7.1.0",
    "ipywidgets>=8.1.8",
    "jax[cuda12]>=0.6.0",
    "flax>=0.10.0",
    "optax>=0.2.4",
    "llvmlite>=0.46.0",
    "matplotlib>=3.10.8",
    "numpy>=2.4.1",
    "orbax-checkpoint>=0.11.33",
    "polars>=1.27.1",
    "rich>=14.0.0",
    "scikit-learn>=1.6.0",
    "seaborn>=0.13.2",
    "shap>=0.49.1",
    "tqdm>=4.66.0",
]

[dependency-groups]
dev = [
    "jupytext>=1.19.1",
    "ruff>=0.15.7",
    "ty>=0.0.12",
]

[build-system]
requires = ["uv_build>=0.11.3,<0.12"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "gp"
namespace = true
```

**Step 2: Install dependencies**

```bash
cd guepard-shield-model && uv sync
```

Expected: deps resolve, JAX detects GPU.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: swap keras for flax+optax dependencies"
```

---

## Task 2: Rewrite Model (`transformer.py`)

**Files:**
- Modify: `src/gp/models/transformer.py`
- Create: `src/gp/models/__init__.py`

**Step 1: Create `src/gp/models/__init__.py`**

```python
from gp.models.transformer import SyscallTransformer
```

**Step 2: Rewrite `transformer.py`**

Replace entire file. Key points:
- `nnx.Module` subclass with `__init__` + `__call__`
- `nnx.Embed` for token embeddings
- Sinusoidal positional encoding as non-trainable `nnx.Variable`
- `nnx.MultiHeadAttention(is_causal=True)` for causal self-attention
- Pre-norm residual blocks (same as Keras version: norm→attn→residual→norm→ffn→residual)
- `nnx.Linear` for FFN (GELU activation via `jax.nn.gelu`)
- `nnx.Dropout` for regularization
- Output head: `nnx.Linear(vocab_size, dtype=jnp.float32)` for stable softmax
- Constructor takes `rngs: nnx.Rngs` parameter

```python
"""Syscall Transformer for Anomaly Detection (Next Token Prediction) — Flax NNX."""

from __future__ import annotations

import math
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


def sinusoidal_pos_enc(max_seq_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
    enc = np.where(dims % 2 == 0, np.sin(angles), np.cos(angles))
    return enc[np.newaxis, :, :].astype(np.float32)


class TransformerEncoderLayer(nnx.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=nhead, in_features=d_model,
            qkv_features=d_model, out_features=d_model,
            decode=False, rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(dropout)
        self.norm2 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.ff1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.ff2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        self.dropout_ff = nnx.Dropout(dropout)
        self.dropout2 = nnx.Dropout(dropout)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, is_causal=True, deterministic=deterministic)
        x = residual + self.dropout1(x, deterministic=deterministic)

        residual = x
        x = self.norm2(x)
        x = self.ff2(self.dropout_ff(jax.nn.gelu(self.ff1(x)), deterministic=deterministic))
        x = residual + self.dropout2(x, deterministic=deterministic)
        return x


class SyscallTransformer(nnx.Module):
    def __init__(self, vocab_size: int, *, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_len: int = 100,
                 rngs: nnx.Rngs | None = None) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_enc = nnx.Variable(
            sinusoidal_pos_enc(max_seq_len, d_model),
        )
        self.input_dropout = nnx.Dropout(dropout)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.norm_out = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, dtype=jnp.float32, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        h = self.embed(x) * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        h = h + self.pos_enc.value[:, :seq_len, :]
        h = self.input_dropout(h, deterministic=deterministic)
        for layer in self.encoder_layers:
            h = layer(h, deterministic=deterministic)
        h = self.norm_out(h)
        return self.head(h)
```

**Step 3: Verify model instantiates and forward-passes**

```python
# Quick smoke test (run from guepard-shield-model/)
from gp.models.transformer import SyscallTransformer
from flax import nnx
import jax

model = SyscallTransformer(vocab_size=100, d_model=64, nhead=4, num_layers=2,
                           dim_feedforward=256, max_seq_len=100, rngs=nnx.Rngs(0))
x = jax.random.randint(jax.random.key(1), (4, 100), 0, 100)
out = model(x, deterministic=False)
print(out.shape)  # (4, 100, 100)
```

**Step 4: Commit**

```bash
git add src/gp/models/transformer.py src/gp/models/__init__.py
git commit -m "feat: rewrite SyscallTransformer as Flax NNX module"
```

---

## Task 3: Rewrite Trainer (`trainer.py`)

**Files:**
- Modify: `src/gp/training/trainer.py`
- Create: `src/gp/training/__init__.py`

This is the biggest task. The new trainer has:
1. Custom training loop (no `model.fit()`)
2. `optax.MultiSteps` for gradient accumulation
3. `optax.cosine_decay_schedule` for LR
4. Manual loss with PAD masking
5. Epoch-level eval with recording-level AUROC
6. Orbax checkpoint save/load
7. Early stopping

**Step 1: Create `src/gp/training/__init__.py`**

```python
```

(Empty init file)

**Step 2: Rewrite `trainer.py`**

```python
"""Training and evaluation for SyscallTransformer — Flax NNX.

Custom training loop with Optax gradient accumulation, CosineDecay LR,
Orbax checkpointing, and per-epoch recording-level AUROC eval.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from gp.config import cfg
from gp.training.metrics import aggregate_ntp_by_recording, compute_metrics


def _ntp_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Cross-entropy loss with PAD (class 0) masking.

    Args:
        logits: [B, W, V] float32.
        targets: [B, W] int32 (PAD=0).

    Returns:
        Scalar mean loss over non-PAD positions.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Gather log-prob at target index → [B, W]
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)
    mask = (targets != 0).astype(jnp.float32)
    return -(target_log_probs * mask).sum() / (mask.sum() + 1e-9)


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Single micro-batch train step. Returns scalar loss."""
    def loss_fn(model: nnx.Module) -> jax.Array:
        logits = model(x, deterministic=False)
        return _ntp_loss(logits, y)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(
    model: nnx.Module,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Eval loss for one batch."""
    logits = model(x, deterministic=True)
    return _ntp_loss(logits, y)


def _forward_logits_batches(
    model: nnx.Module,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Forward pass in batches, returning softmax probabilities [N, W, V]."""
    n = len(X)
    all_probs: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch_x = jnp.array(X[start : start + batch_size])
        logits = model(batch_x, deterministic=True)
        probs = np.array(jax.nn.softmax(logits, axis=-1))
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def compute_ntp_scores(
    model: nnx.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Compute per-window NLL anomaly scores for NTP model."""
    bs = batch_size or cfg.eval_batch_size
    probs = _forward_logits_batches(model, X, bs)
    b, w = X.shape[0], X.shape[1]
    row_idx = np.arange(b)[:, None]
    col_idx = np.arange(y.shape[1])
    true_probs = probs[row_idx, col_idx, y]
    nll = -np.log(true_probs + 1e-9)
    mask = (y != 0).astype(np.float32)
    return (nll * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-9)


def compute_ntp_per_position(
    model: nnx.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Compute per-position NLL for NTP model. NaN at PAD positions."""
    bs = batch_size or cfg.eval_batch_size
    probs = _forward_logits_batches(model, X, bs)
    b = X.shape[0]
    row_idx = np.arange(b)[:, None]
    col_idx = np.arange(y.shape[1])
    true_probs = probs[row_idx, col_idx, y]
    nll = -np.log(true_probs + 1e-9)
    pad_mask = y == 0
    nll[pad_mask] = np.nan
    return nll


def train_loop(
    model: nnx.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_labels: dict[int, int],
    val_rec_ids: np.ndarray,
) -> dict:
    """Full training loop with gradient accumulation, early stopping, checkpointing."""
    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = len(X_train)
    bs = cfg.batch_size
    eff_bs = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = n_train // eff_bs
    micro_batches_per_step = cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs

    # Optimizer with gradient accumulation
    lr_schedule = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate,
        decay_steps=total_steps,
        alpha=0.1,
    )
    base_opt = optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    opt = optax.MultiSteps(base_opt, every_k_schedule=micro_batches_per_step)
    optimizer = nnx.Optimizer(model, opt)

    # Early stopping state
    best_val_loss = float("inf")
    best_graphdef = None
    best_state = None
    patience_counter = 0

    checkpointer = ocp.StandardCheckpointer()

    print(f"  bs={bs}, grad_accum={cfg.grad_accum_steps}, "
          f"effective_bs={eff_bs}, total_steps={total_steps}")

    for epoch in range(cfg.epochs):
        # Shuffle train data
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        # Train
        model.train()  # not strictly needed for nnx but good practice
        epoch_losses = []
        t0 = time.time()

        for step in range(steps_per_epoch):
            step_loss = 0.0
            for micro in range(micro_batches_per_step):
                idx = step * eff_bs + micro * bs
                x_batch = jnp.array(X_shuf[idx : idx + bs])
                y_batch = jnp.array(y_shuf[idx : idx + bs])
                loss = train_step(model, optimizer, x_batch, y_batch)
                step_loss += float(loss)

            epoch_losses.append(step_loss / micro_batches_per_step)

        avg_train_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0

        # Eval
        model.eval()  # sets deterministic=True behavior conceptually
        val_losses = []
        for start in range(0, len(X_val), cfg.eval_batch_size):
            x_v = jnp.array(X_val[start : start + cfg.eval_batch_size])
            y_v = jnp.array(y_val[start : start + cfg.eval_batch_size])
            val_losses.append(float(eval_step(model, x_v, y_v)))
        avg_val_loss = np.mean(val_losses)

        # Recording-level AUROC
        val_scores = compute_ntp_scores(model, X_val, y_val)
        y_true_rec, y_score_rec = aggregate_ntp_by_recording(
            val_scores, val_rec_ids, val_labels
        )
        m = compute_metrics(
            y_true_rec, (y_score_rec > y_score_rec.mean()).astype(int), y_score_rec
        )
        val_auroc = m["auroc"] if not np.isnan(m["auroc"]) else 0.0
        val_f1 = m["f1"]

        print(f"Epoch {epoch+1}/{cfg.epochs} — "
              f"train_loss: {avg_train_loss:.4f} val_loss: {avg_val_loss:.4f} "
              f"val_auroc: {val_auroc:.4f} val_f1: {val_f1:.4f} "
              f"time: {elapsed:.1f}s")

        # Checkpoint & early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            graphdef, state = nnx.split(model)
            best_graphdef = graphdef
            best_state = state
            # Save Orbax checkpoint
            checkpointer.save(
                cfg.ckpt_path / "checkpoint",
                state,
                overwrite=True,
                force=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_graphdef is not None and best_state is not None:
        model = nnx.merge(best_graphdef, best_state)

    return {"best_val_loss": float(best_val_loss)}


def batch_predict(
    model: nnx.Module, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window NLL anomaly scores for a data split."""
    X_raw = np.load(cfg.npy_dir / f"{split}_X.npy", mmap_mode="r")
    rec_ids = np.load(cfg.npy_dir / f"{split}_rec_ids.npy")
    X = X_raw[:, :-1]
    y = X_raw[:, 1:]
    scores = compute_ntp_scores(model, X, y)
    return scores, rec_ids


def load_best_checkpoint(model: nnx.Module) -> nnx.Module:
    """Load best Orbax checkpoint into model."""
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(
        lambda: SyscallTransformer(
            vocab_size=model.vocab_size,
            d_model=model.d_model,
            nhead=model.nhead if hasattr(model, 'nhead') else cfg.nhead,
            num_layers=len(model.encoder_layers) if hasattr(model, 'encoder_layers') else cfg.num_layers,
            dim_feedforward=model.ff1.out_features if hasattr(model, 'ff1') else cfg.dim_feedforward,
            dropout=0.0,
            max_seq_len=model.max_seq_len,
        )
    )
    _, abstract_state = nnx.split(abstract_model)
    restored_state = checkpointer.restore(cfg.ckpt_path / "checkpoint", abstract_state)
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, restored_state)
    return model
```

**Step 3: Commit**

```bash
git add src/gp/training/trainer.py src/gp/training/__init__.py
git commit -m "feat: rewrite trainer with Flax NNX custom loop, Optax grad accum, Orbax ckpt"
```

---

## Task 4: Update Config (`config.py`)

**Files:**
- Modify: `src/gp/config.py`

**Step 1: Update config**

Remove `use_mixed_precision` (now handled by JAX matmul precision env). Keep everything else unchanged.

```python
    grad_accum_steps: int = 8  # gradient accumulation: effective bs = 16*8 = 128
    # No more use_mixed_precision — JAX bfloat16 matmul is set via env var
    learning_rate: float = 1e-3
```

Also add these new fields:

```python
    # ── Mixed precision (JAX) ────────────────────────────────────────────
    use_mixed_precision: bool = True  # sets XLA_FLAGS for bfloat16 matmul
```

Actually, let me reconsider. The `use_mixed_precision` flag is still useful as a toggle. We'll use it in the notebook to set `XLA_FLAGS` before JAX init rather than in config. Let's keep it but change the comment.

**Step 2: Commit**

```bash
git add src/gp/config.py
git commit -m "refactor: update config comments for Flax (no functional change)"
```

---

## Task 5: Update Evaluation (`extractor.py`)

**Files:**
- Modify: `src/gp/evaluation/extractor.py`

The `extractor.py` calls `compute_ntp_per_position` with a Keras model. After migration, it receives an `nnx.Module`. The function signature stays the same — only the type annotation changes from `keras.Model` to `nnx.Module`. Since `compute_ntp_per_position` was moved to work with Flax NNX, `extractor.py` should just work.

No changes needed to `extractor.py` itself, but the import path may change.

**Step 1: Verify extractor.py still works with new imports**

No changes needed — it already imports `compute_ntp_per_position` from `gp.training.trainer`.

**Step 2: Commit (if changes were made)**

```bash
git add src/gp/evaluation/extractor.py
git commit -m "refactor: update type hints in extractor for Flax NNX"
```

---

## Task 6: Rewrite Notebooks

**Files:**
- Modify: `notebooks/p2_transformer_teacher.py`
- Modify: `notebooks/p2_pilot_test.py`
- Modify: `notebooks/p3_transition_analysis.py`

### 6a: p2_transformer_teacher.py

Replace Keras imports/setup with Flax NNX. The notebook now calls `train_loop()` directly.

```python
# %% [markdown]
# # P2: Transformer Teacher Training (Next Token Prediction) — Flax NNX

# %% [1. Imports]
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if os.environ.get("GP_USE_MIXED_PRECISION", "1") == "1":
    os.environ.setdefault("XLA_FLAGS", "--xla_default_matmul_precision=bfloat16")

import numpy as np
from flax import nnx
from gp.config import cfg
from gp.models.transformer import SyscallTransformer
from gp.training.metrics import compute_metrics
from gp.training.trainer import batch_predict, train_loop
from rich.console import Console

console = Console()

# %% [2. Load Metadata]
with open(cfg.npy_dir / "train_stats.json") as f:
    train_meta = json.load(f)
with open(cfg.npy_dir / "val_stats.json") as f:
    val_meta = json.load(f)
with open(cfg.npy_dir / "test_stats.json") as f:
    test_meta = json.load(f)

vocab_size = train_meta["vocab_size"]

# %% [3. Build Model]
model = SyscallTransformer(
    vocab_size=vocab_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_seq_len=cfg.window_size,
    rngs=nnx.Rngs(0),
)

# %% [4. Load data (mmap)
X_train_mmap = np.load(cfg.npy_dir / "train_X.npy", mmap_mode="r")
X_val_mmap = np.load(cfg.npy_dir / "val_X.npy", mmap_mode="r")
rec_ids_val = np.load(cfg.npy_dir / "val_rec_ids.npy")
y_val_labels = {r["rec_id"]: r["is_exploit"] for r in val_meta["recordings"]}

X_train = X_train_mmap[:, :-1]
y_train = X_train_mmap[:, 1:]
X_val = X_val_mmap[:, :-1]
y_val = X_val_mmap[:, 1:]

# %% [5. Train]
console.print("\n[bold yellow]Training...[/bold yellow]")
result = train_loop(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    val_labels=y_val_labels,
    val_rec_ids=rec_ids_val,
)

# %% [6. Evaluate on Test Set]
console.print("\n[bold green]Evaluating on Test Set...[/bold green]")
test_scores, test_rec_ids = batch_predict(model, "test")

test_rec_labels = {r["rec_id"]: r["is_exploit"] for r in test_meta["recordings"]}
unique_ids = np.unique(test_rec_ids)
y_true = np.array([test_rec_labels[rid] for rid in unique_ids])
y_score = np.array([test_scores[test_rec_ids == rid].max() for rid in unique_ids])

val_scores, _ = batch_predict(model, "val")
threshold = np.percentile(val_scores, 95)
y_pred = (y_score > threshold).astype(int)

test_metrics = compute_metrics(y_true, y_pred, y_score)
console.print("\n[bold]Test set metrics:[/bold]")
for k, v in test_metrics.items():
    console.print(f"  {k:12s}: {v:.4f}")

# %% [7. Per-scenario]
rec_id_to_scenario = {r["rec_id"]: r["scenario"] for r in test_meta["recordings"]}
scenarios = sorted({r["scenario"] for r in test_meta["recordings"]})

from polars import DataFrame
sc_results = []
for sc in scenarios:
    sc_rec_ids = {rid for rid, s in rec_id_to_scenario.items() if s == sc}
    mask = np.isin(unique_ids, list(sc_rec_ids))
    if not mask.any():
        continue
    m = compute_metrics(y_true[mask], y_pred[mask], y_score[mask])
    sc_results.append({"scenario": sc, **m})

console.print(DataFrame(sc_results))
```

### 6b: p2_pilot_test.py

```python
# %% [markdown]
# # P2 Pilot: Quick pipeline verification with subsampled data — Flax NNX

# %% [1. Imports & Env]
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_FLAGS", "--xla_default_matmul_precision=bfloat16")

import numpy as np
from flax import nnx
from gp.config import cfg
from gp.models.transformer import SyscallTransformer
from gp.training.trainer import compute_ntp_scores, train_loop
from rich.console import Console

console = Console()

# %% [2. Override config for pilot]
cfg.epochs = 2
cfg.batch_size = 32
cfg.eval_batch_size = 32
cfg.patience = 2
cfg.grad_accum_steps = 2  # small for pilot

cfg.d_model = 64
cfg.nhead = 4
cfg.num_layers = 2
cfg.dim_feedforward = 256

PILOT_N_TRAIN = 500
PILOT_N_VAL = 100

# %% [3. Load & subsample data]
console.print("[bold cyan]Loading and subsampling data...[/bold cyan]")

X_train_raw = np.load(cfg.npy_dir / "train_X.npy")
X_val_raw = np.load(cfg.npy_dir / "val_X.npy")
rec_ids_val = np.load(cfg.npy_dir / "val_rec_ids.npy")

with open(cfg.npy_dir / "val_stats.json") as f:
    val_meta = json.load(f)
y_val_labels = {r["rec_id"]: r["is_exploit"] for r in val_meta["recordings"]}

rng = np.random.default_rng(42)
train_idx = rng.choice(len(X_train_raw), size=min(PILOT_N_TRAIN, len(X_train_raw)), replace=False)
val_idx = rng.choice(len(X_val_raw), size=min(PILOT_N_VAL, len(X_val_raw)), replace=False)
train_idx.sort()
val_idx.sort()

X_train = X_train_raw[train_idx, :-1][:, :256]
y_train = X_train_raw[train_idx, 1:][:, :256]
X_val = X_val_raw[val_idx, :-1][:, :256]
y_val = X_val_raw[val_idx, 1:][:, :256]
rec_ids_sub = rec_ids_val[val_idx]

console.print(f"  train: {X_train.shape}, val: {X_val.shape}")

with open(cfg.npy_dir / "train_stats.json") as f:
    train_meta = json.load(f)

vocab_size = train_meta["vocab_size"]

# %% [4. Build & train]
model = SyscallTransformer(
    vocab_size=vocab_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
    max_seq_len=256,
    rngs=nnx.Rngs(0),
)

console.print("\n[bold yellow]Pilot Training (2 epochs)...[/bold yellow]")
result = train_loop(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    val_labels=y_val_labels,
    val_rec_ids=rec_ids_sub,
)

# %% [5. Evaluate anomaly scores on subsampled val]
console.print("\n[bold green]Pilot Evaluation on val subsample...[/bold green]")
val_scores = compute_ntp_scores(model, X_val, y_val)
threshold = np.percentile(val_scores, 95)
console.print(f"  Threshold (val p95): {threshold:.4f}")
console.print(f"  Mean val NLL: {val_scores.mean():.4f}")
console.print(f"  Std  val NLL: {val_scores.std():.4f}")

console.print("\n[bold green]Pilot run complete! Pipeline verified.[/bold green]")
```

### 6c: p3_transition_analysis.py

```python
# %% [markdown]
# # P3 Pilot: Transition-Level Rule Candidate Extraction — Flax NNX

# %% [1. Setup]
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_FLAGS", "--xla_default_matmul_precision=bfloat16")

import numpy as np
from flax import nnx
from gp.config import cfg
from gp.evaluation.extractor import (
    compute_transition_matrix,
    extract_transition_attributions,
)
from gp.models.transformer import SyscallTransformer
from gp.training.metrics import aggregate_ntp_by_recording, compute_metrics
from gp.training.trainer import compute_ntp_per_position, compute_ntp_scores, load_best_checkpoint
from rich.console import Console
from rich.table import Table

console = Console()

# %% [2. Load model & data]
with open(cfg.vocab_path) as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

model = SyscallTransformer(
    vocab_size=len(vocab),
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=0.0,
    max_seq_len=cfg.window_size,
    rngs=nnx.Rngs(0),
)
model = load_best_checkpoint(model)

X_test = np.load(cfg.npy_dir / "test_X.npy", mmap_mode="r")
X_test_input = X_test[:, :-1]
y_test = X_test[:, 1:]
rec_ids = np.load(cfg.npy_dir / "test_rec_ids.npy")

with open(cfg.npy_dir / "test_stats.json") as f:
    test_meta = json.load(f)
test_labels = {r["rec_id"]: r["is_exploit"] for r in test_meta["recordings"]}

# %% [3. Baseline: Normal bigram transition matrix]
console.print("[bold]Computing baseline transition matrix from train (normal-only)...[/bold]")
P_baseline = compute_transition_matrix(split="train")
console.print(f"  Shape: {P_baseline.shape}")

# %% [4. Per-position NLL extraction]
console.print("[bold]Computing per-position NLL on test set...[/bold]")
per_pos_nll = compute_ntp_per_position(model, X_test_input, y_test)
window_scores = np.nanmean(per_pos_nll, axis=1)
console.print(f"  per_pos_nll shape: {per_pos_nll.shape}")
console.print(f"  Mean NLL (non-PAD): {np.nanmean(per_pos_nll):.4f}")

# %% [5. Threshold & recording-level aggregation]
threshold = np.percentile(window_scores, 95)
y_true_rec, y_score_rec = aggregate_ntp_by_recording(
    window_scores, rec_ids, test_labels
)
y_pred_rec = (y_score_rec > threshold).astype(int)
m = compute_metrics(y_true_rec, y_pred_rec, y_score_rec)

console.print("\n[bold]Recording-level metrics:[/bold]")
for k, v in m.items():
    console.print(f"  {k}: {v:.4f}")

# %% [6. Transition attributions]
console.print("\n[bold]Extracting transition attributions...[/bold]")
attribution = extract_transition_attributions(model, split="test", top_k=20)

exploit_transitions: dict[tuple[str, str], float] = {}
for attr in attribution:
    if not attr.is_exploit:
        continue
    for (src, dst), nll in zip(attr.top_transition_ids, attr.top_transition_nlls):
        key = (inv_vocab.get(src, f"unk({src})"), inv_vocab.get(dst, f"unk({dst})"))
        if key not in exploit_transitions or nll > exploit_transitions[key]:
            exploit_transitions[key] = float(nll)

sorted_trans = sorted(exploit_transitions.items(), key=lambda x: -x[1])[:20]

table = Table(title="Top 20 Anomalous Transitions (Exploit Recordings)")
table.add_column("Transition", style="cyan")
table.add_column("Max NLL", justify="right")
table.add_column("Baseline P(next|current)", justify="right")
for (src_name, dst_name), nll in sorted_trans:
    src_id = vocab.get(src_name, 1)
    dst_id = vocab.get(dst_name, 1)
    baseline_p = P_baseline[src_id, dst_id] if src_id < len(P_baseline) else 0
    table.add_row(f"{src_name} -> {dst_name}", f"{nll:.3f}", f"{baseline_p:.6f}")
console.print(table)

# %% [7. Per-scenario top transitions]
console.print("\n[bold]Per-scenario top transitions:[/bold]")
scenarios = sorted(set(attr.scenario for attr in attribution if attr.is_exploit))
for sc in scenarios[:5]:
    sc_attrs = [attr for attr in attribution if attr.scenario == sc and attr.is_exploit]
    sc_trans: dict[tuple[int, int], float] = {}
    for attr in sc_attrs:
        for (s, d), nll in zip(attr.top_transition_ids, attr.top_transition_nlls):
            key = (s, d)
            if key not in sc_trans or nll > sc_trans[key]:
                sc_trans[key] = float(nll)
    top3 = sorted(sc_trans.items(), key=lambda x: -x[1])[:3]
    console.print(f"\n  [yellow]{sc}[/yellow]")
    for (s, d), nll in top3:
        s_name = inv_vocab.get(s, f"unk({s})")
        d_name = inv_vocab.get(d, f"unk({d})")
        baseline_p = P_baseline[s, d] if s < len(P_baseline) else 0
        console.print(f"    {s_name} -> {d_name}  NLL={nll:.3f}  baseline_P={baseline_p:.6f}")
```

**Step 2: Commit**

```bash
git add notebooks/p2_transformer_teacher.py notebooks/p2_pilot_test.py notebooks/p3_transition_analysis.py
git commit -m "feat: rewrite notebooks for Flax NNX training loop"
```

---

## Task 7: Update WALKTHROUGH.md

**Files:**
- Modify: `WALKTHROUGH.md`

Update to reflect Flax NNX stack:
- Model section → `nnx.Module` with `nnx.MultiHeadAttention(is_causal=True)`
- Loss → manual cross-entropy with PAD mask
- Optimizer → `optax.adamw` + `optax.cosine_decay_schedule` + `optax.MultiSteps`
- Checkpoint → Orbax `StandardCheckpointer`
- Mixed precision → `XLA_FLAGS=--xla_default_matmul_precision=bfloat16`
- Gradient accumulation → actually works now via `optax.MultiSteps`

**Step 1: Commit**

```bash
git add WALKTHROUGH.md
git commit -m "docs: update WALKTHROUGH for Flax NNX migration"
```

---

## Task 8: Pilot Test — Verify Pipeline

**Step 1: Run pilot**

```bash
cd guepard-shield-model && uv run python notebooks/p2_pilot_test.py
```

Expected: training completes 2 epochs, val NLL decreases, no OOM on 6GB VRAM.

**Step 2: Fix any issues found**

Common gotchas to watch for:
- `nnx.MultiHeadAttention` API differences (is_causal parameter)
- Shape mismatches in loss computation
- JAX device placement issues (ensure arrays on GPU)
- Sinusoidal pos encoding slicing
- Mixed precision behavior with `nnx.Linear(dtype=jnp.float32)` head

**Step 3: Commit fixes (if any)**

```bash
git add -A && git commit -m "fix: address pilot test issues"
```

---

## Task 9: Update `metrics.py` and `extractor.py` Type Hints

**Files:**
- Modify: `src/gp/training/metrics.py`
- Modify: `src/gp/evaluation/extractor.py`

Change any `keras.Model` type hints to `nnx.Module` or remove them (the functions already work with any model that has a `__call__` method).

Actually, looking at `metrics.py` — it has no Keras imports. It's pure numpy/sklearn. No changes needed.

For `extractor.py` — the type hint for `model` parameter is untyped. We can add `nnx.Module` as a type hint or leave it as-is since it just calls `compute_ntp_per_position` which accepts any model.

Leave as-is for now.

---

## Summary of Changes

| File | Change |
|---|---|
| `pyproject.toml` | Remove `keras`, add `flax`, `optax` |
| `src/gp/models/__init__.py` | New: export `SyscallTransformer` |
| `src/gp/models/transformer.py` | Full rewrite: Keras → Flax NNX |
| `src/gp/training/__init__.py` | New: empty init |
| `src/gp/training/trainer.py` | Full rewrite: Keras fit → custom loop with Optax |
| `src/gp/training/metrics.py` | No changes (pure numpy) |
| `src/gp/evaluation/extractor.py` | Minimal — update model type hints if desired |
| `src/gp/config.py` | Minimal — update comments |
| `notebooks/p2_transformer_teacher.py` | Full rewrite: Flax NNX training |
| `notebooks/p2_pilot_test.py` | Full rewrite: Flax NNX pilot |
| `notebooks/p3_transition_analysis.py` | Full rewrite: Flax NNX eval |
| `WALKTHROUGH.md` | Update docs |

**Key benefits of this migration:**
1. **Gradient accumulation works natively** via `optax.MultiSteps`
2. **Explicit training loop** — no more Keras callback hacks
3. **Mixed precision** is just an XLA flag, not a global Keras policy
4. **Orbax checkpoint** is direct `nnx.split/merge`, not through a Keras callback
5. **No more `ignore_class=0`** — manual PAD masking is transparent and debuggable