from pathlib import Path
from typing import Type

import lightning as L
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..config import TeacherConfig, WindowConfig
from ..data_loader.datamodule import TeacherDataModule
from ..data_loader.windowing import num_sliding_windows
from .callbacks import DatasetReshuffleCallback, MetricsHistory
from .teacher_module import TeacherLightningModule


def compute_class_weights(
    train_metas: list,
    window_config: WindowConfig,
    max_windows_per_seq: int | None,
    post_exploit_fraction: float = 0.5,
) -> list[float]:
    """Estimate [w_normal, w_attack] from training split window counts.

    Attack recordings contain ~post_exploit_fraction of windows that are truly
    attack (post-exploit). Uses sqrt(n_normal / n_attack) to dampen extreme ratios.
    """

    def capped(m) -> int:
        n = num_sliding_windows(m.seq_length, window_config)
        return min(n, max_windows_per_seq) if max_windows_per_seq else n

    n_normal = sum(capped(m) for m in train_metas if m.label == 0)
    n_attack = max(
        int(
            sum(capped(m) for m in train_metas if m.label == 1) * post_exploit_fraction
        ),
        1,
    )
    attack_weight = (n_normal / n_attack) ** 0.5
    return [1.0, float(attack_weight)]


def train_teacher(
    model_cls: Type,
    model_kwargs: dict,
    config: TeacherConfig,
    datamodule: TeacherDataModule,
    output_dir: Path,
    tag: str,
    max_epochs: int,
    patience: int,
    precision: _PRECISION_INPUT,
) -> tuple[TeacherLightningModule, MetricsHistory]:
    """Train a teacher model, reload best checkpoint, return module + history."""
    model = model_cls(**model_kwargs)
    module = TeacherLightningModule(model, config)
    history = MetricsHistory()

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"teacher_{tag}",
        monitor="val_f1",
        save_top_k=1,
        mode="max",
    )
    callbacks: list[L.Callback] = [
        ckpt,
        EarlyStopping(monitor="val_f1", patience=patience, mode="max"),
        DatasetReshuffleCallback(datamodule),
        LearningRateMonitor(logging_interval="epoch"),
        history,
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=True,
    )

    print(f"\n{'=' * 60}\nTraining {tag} (precision={precision})\n{'=' * 60}")
    trainer.fit(module, datamodule)

    if ckpt.best_model_path:
        module = TeacherLightningModule.load_from_checkpoint(
            ckpt.best_model_path, model=model_cls(**model_kwargs), config=config
        )

    return module, history
