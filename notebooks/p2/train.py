"""Phase 2 training script.

Run from project root:
    python notebooks/p2/train.py

Reads preprocessed .npy files from data/processed/p2/ and trains
SyscallTransformer with cosine-decay + warmup. Best checkpoint saved
to results/p2/checkpoints/.
"""

from __future__ import annotations

import json
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gp.datamodule import SyscallDataModule
from gp.model import SyscallTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR   = PROJECT_ROOT / "data" / "processed" / "p2"
CKPT_DIR   = PROJECT_ROOT / "results" / "p2" / "checkpoints"
VOCAB_PATH = DATA_DIR / "vocab.json"

BATCH_SIZE   = 256
MAX_EPOCHS   = 20
NUM_WORKERS  = 0


def main() -> None:
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"vocab_size: {vocab_size}")

    dm = SyscallDataModule(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    dm.setup("fit")

    # Estimate total steps for cosine schedule
    steps_per_epoch = len(dm.train_dataloader())
    max_steps = MAX_EPOCHS * steps_per_epoch

    model = SyscallTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        window_size=64,
        lr=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_steps=max_steps,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        precision="16-mixed",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        default_root_dir=str(CKPT_DIR),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(
                dirpath=str(CKPT_DIR),
                monitor="val_loss",
                save_top_k=1,
                filename="best",
            ),
        ],
    )
    trainer.fit(model, dm)
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
