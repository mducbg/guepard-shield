# %% [markdown]
# # Phase 2: Train Syscall Transformer (LID-DS-2021)
# 
# This notebook trains a Causal Transformer for next-token prediction
# using PyTorch Lightning.

# %%
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from gp.config import (
    d_model, nhead, dim_feedforward, dropout, 
    learning_rate, ckpt_path, use_mixed_precision
)
from gp.model.transformer import SyscallTransformer
from gp.data_loader.lidds_2021_torch import SyscallDataModule

# %%
# Hyperparameters
batch_size = 64 # Physical batch for RTX 3060 6GB
grad_accum = 2  # effective batch size = 128
max_epochs = 30 

# %%
# Initialize Model
model = SyscallTransformer(
    vocab_size=102, # 99 real + <unk> + <unknown> + <pad>
    d_model=d_model,
    nhead=nhead,
    num_layers=4,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    lr=learning_rate,
    pad_idx=0
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_path,
    filename="best-transformer-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

# Logger
logger = CSVLogger("results/logs", name="transformer")

# %%
# Trainer Execution
if __name__ == "__main__":
    # Optimize matmul for RTX 3060
    torch.set_float32_matmul_precision('high')
    
    # Initialize DataModule with subsampling (50 windows/file).
    # Note: Subsampling acts as an implicit regularizer that improves anomaly
    # detection despite higher validation loss (see WALKTHROUGH.md).
    dm = SyscallDataModule(batch_size=batch_size, max_windows_train=50)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=grad_accum,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision="16-mixed" if use_mixed_precision else 32
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print(f"Training finished. Best model saved at: {checkpoint_callback.best_model_path}")
