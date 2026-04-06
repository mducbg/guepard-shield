from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WindowConfig:
    window_size: int = 64
    stride: int = 12
    min_window_size: int = 16


@dataclass
class VocabConfig:
    min_freq: int = 1
    max_size: int = 512


@dataclass
class TeacherConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 1000
    pooling: str = "cls"  # "cls" | "mean"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    temperature: float = 1.0
    # Per-class loss weights [w_normal, w_attack]. None = uniform.
    # Use sqrt(n_normal / n_attack) for attack weight to counter imbalance
    # without over-penalising normal windows.
    class_weights: list[float] | None = None
    # Linear warmup epochs before cosine decay kicks in.
    warmup_epochs: int = 5


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    window: WindowConfig = field(default_factory=WindowConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    num_workers: int = 4
    batch_size: int = 64
