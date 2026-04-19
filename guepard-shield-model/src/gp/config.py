"""Configuration module for guepard-shield-model.

Defines data paths and a single mutable ``cfg`` object that holds all
hyperparameters. Notebooks import ``cfg`` and override fields as needed;
library functions read from ``cfg`` directly so callers don't pass long
keyword-argument lists.

Example (notebook)::

    from gp.config import cfg
    cfg.d_model = 512        # override before training
    cfg.num_layers = 8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Base paths
# The data directory is inside guepard-shield-model/, not at project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_ROOT = Path(__file__).parent.parent.parent  # guepard-shield-model/
DATA_ROOT = MODEL_ROOT / "data"

# Extracted data paths
EXTRACTED_DATA_DIR = DATA_ROOT / "extracted"
DONGTING_DIR = EXTRACTED_DATA_DIR / "DongTing"
LIDDS_2019_DIR = EXTRACTED_DATA_DIR / "LID-DS-2019"
LIDDS_2021_DIR = EXTRACTED_DATA_DIR / "LID-DS-2021"

# Processed data paths (ArrayRecord files, gitignored)
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

# Results
RESULTS_DIR = MODEL_ROOT / "results"

# Splits directory
SPLITS_DIR = DATA_ROOT / "splits"

# Syscall table
TBL_PATH = DONGTING_DIR / "syscall_64.tbl"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All tunable hyperparameters in one place.

    Import the singleton ``cfg`` and mutate fields before calling any
    training or data-pipeline function.
    """

    # ── Data pipeline ────────────────────────────────────────────────────
    window_size: int = (
        1000  # syscalls per window; EDA shows ~1K needed to capture attack context
    )
    stride_train: int = 500  # 50% overlap for train/val
    stride_eval: int = 1000  # non-overlapping for test (correct aggregation)
    max_windows_train: int = (
        500  # cap per recording in train split (keeps disk use bounded)
    )
    max_windows_eval: int | None = (
        None  # None = all windows (required for correct aggregation)
    )
    max_syscalls_train: int = (
        251_000  # = max_windows_train * stride_train + window_size
    )
    max_syscalls_eval: int = 50_000
    vocab_min_freq: int = 2

    # ── Model architecture ───────────────────────────────────────────────
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 16  # physical batch; effective = batch_size * grad_accum_steps
    eval_batch_size: int = 16
    grad_accum_steps: int = 8  # gradient accumulation: effective bs = 16*8 = 128
    use_mixed_precision: bool = True  # bfloat16 halves VRAM for attention
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    threshold: float = 0.5

    # ── Paths ────────────────────────────────────────────────────────────
    ar_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR / "lidds2021_ar")
    npy_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR / "lidds2021_npy")
    vocab_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "vocab_transformer.json"
    )
    ckpt_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "checkpoints" / "transformer" / "best"
    )
    metrics_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "p2_transformer_metrics.json"
    )
    history_plot_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "p2_train_history.png"
    )


cfg = Config()
