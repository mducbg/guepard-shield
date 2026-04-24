# %% [markdown]
# # Preprocess LID-DS-2021 (Train/Val only)
# 
# This script specifically processes the training and validation splits of LID-DS-2021.
# It applies deduplication and stride=1000 for maximum efficiency.

# %%
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gp.config import LIDDS_2021_DIR, PROCESSED_DATA_DIR
from gp.data_loader.vocabulary import SyscallVocabulary

# %%
# 1. Setup Vocabulary and Paths
vocab_file = Path("results/eda_cross_dataset/vocab_lidds2021_train.txt")
if not vocab_file.exists():
    vocab_file = Path("guepard-shield-model") / vocab_file

print(f"Loading vocabulary from {vocab_file}")
vocab = SyscallVocabulary.from_file(vocab_file)

output_base_dir = PROCESSED_DATA_DIR / "lidds2021"
output_base_dir.mkdir(parents=True, exist_ok=True)

# %%
# 2. Identify target files for Train and Val
# Structure: LID-DS-2021/<scenario>/(training|validation)/<rec_id>/<rec_id>.sc

def get_sc_files(root: Path, split_name: str):
    """split_name: 'training' or 'validation'"""
    pattern = f"*/{split_name}/*/*.sc"
    return sorted(list(root.glob(pattern)))

train_files = get_sc_files(LIDDS_2021_DIR, "training")
val_files = get_sc_files(LIDDS_2021_DIR, "validation")

all_targets = []
for f in train_files: all_targets.append((f, "train"))
for f in val_files: all_targets.append((f, "val"))

print(f"Found {len(train_files)} training recordings and {len(val_files)} validation recordings.")

# %%
# 3. Processing Function

def process_file(sc_path: Path, split: str):
    syscall_names = []
    with open(sc_path, "r") as f:
        for line in f:
            parts = line.split(maxsplit=7)
            if len(parts) < 7:
                continue
            # parts[5] is syscall, parts[6] is direction
            if parts[6] == "<":
                syscall_names.append(parts[5])
    
    if not syscall_names:
        return

    # Encode to IDs
    token_ids = vocab.encode(syscall_names)
    token_ids = np.array(token_ids, dtype=np.uint16)
    
    # Windowing strategy: stride 1000, window 1000
    window_size = 1000
    stride = 1000
    
    windows = []
    for i in range(0, len(token_ids) - window_size + 1, stride):
        windows.append(token_ids[i : i + window_size])
        
    if not windows and len(token_ids) < window_size:
        # Pad short sequences
        pad_len = window_size - len(token_ids)
        window = np.pad(token_ids, (0, pad_len), constant_values=0)
        windows.append(window)
    
    if windows:
        windows_array = np.stack(windows)
        
        # Exact Deduplication for Train
        if split == "train":
            windows_array = np.unique(windows_array, axis=0)
        
        # Save
        scenario = sc_path.parent.parent.parent.name
        target_dir = output_base_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # All train/val are normal in LID-DS-2021
        file_name = f"{scenario}_{sc_path.stem}_normal.npy"
        np.save(target_dir / file_name, windows_array)

# %%
# 4. Run loop with correct tqdm total
print("Processing recordings...")
for sc_path, split in tqdm(all_targets, desc="Preprocessing"):
    process_file(sc_path, split)

print("\n--- Preprocessing Complete for Train and Val! ---")
