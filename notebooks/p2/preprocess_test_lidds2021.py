# %% [markdown]
# # Preprocess LID-DS-2021 (Test Set with Window-level Labels)
# 
# This script processes the Test split and uses the 'exploit' timestamp 
# from JSON to label EACH window as either Normal (0) or Attack (1).
# This is critical for high-quality Rule Extraction in Phase 3.

# %%
import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gp.config import LIDDS_2021_DIR, PROCESSED_DATA_DIR
from gp.data_loader.vocabulary import SyscallVocabulary

# %%
# 1. Setup
vocab_file = Path("results/eda_cross_dataset/vocab_lidds2021_train.txt")
if not vocab_file.exists():
    vocab_file = Path("guepard-shield-model") / vocab_file

vocab = SyscallVocabulary.from_file(vocab_file)
output_base_dir = PROCESSED_DATA_DIR / "lidds2021" / "test"
output_base_dir.mkdir(parents=True, exist_ok=True)

# %%
# 2. Identify target files
pattern = "*/test/*/*/*.sc"
all_test_sc = sorted(list(LIDDS_2021_DIR.glob(pattern)))

# %%
# 3. Processing Function with Timestamp Labeling

def process_test_file_with_timestamps(sc_path: Path):
    # Load JSON metadata for exploit timestamp
    json_path = sc_path.with_suffix(".json")
    exploit_ts_sec = None
    if json_path.exists():
        with open(json_path, "r") as j:
            meta = json.load(j)
            if meta.get("exploit", False):
                # Get the first exploit absolute timestamp
                exploits = meta.get("time", {}).get("exploit", [])
                if exploits:
                    exploit_ts_sec = exploits[0].get("absolute")

    syscall_names = []
    syscall_timestamps = [] # Store TS for each syscall
    
    with open(sc_path, "r") as f:
        for line in f:
            parts = line.split(maxsplit=7)
            if len(parts) < 7: continue
            if parts[6] == "<":
                syscall_names.append(parts[5])
                # Convert 19-digit ns to float seconds
                ts_ns = int(parts[0])
                syscall_timestamps.append(ts_ns / 1e9)
    
    if not syscall_names: return

    token_ids = np.array(vocab.encode(syscall_names), dtype=np.uint16)
    ts_array = np.array(syscall_timestamps)
    
    window_size = 1000
    # Stride controls overlap. Use 1000 for fast eval, 200 for finer granularity
    # (catches attacks near window boundaries but increases data volume 5×).
    stride = 200
    
    windows = []
    labels = []
    
    for i in range(0, len(token_ids) - window_size + 1, stride):
        window = token_ids[i : i + window_size]
        windows.append(window)
        
        # Labeling logic:
        # A window is 'Attack' (1) if its END timestamp is after the exploit start
        window_end_ts = ts_array[i + window_size - 1]
        
        is_attack = 0
        if exploit_ts_sec is not None and window_end_ts >= exploit_ts_sec:
            is_attack = 1
        labels.append(is_attack)
        
    if windows:
        windows_np = np.stack(windows)
        labels_np = np.array(labels, dtype=np.int8)
        
        scenario = sc_path.parent.parent.parent.parent.name
        # Final label in filename is for compatibility with simple evaluators
        overall_label = "exploit" if exploit_ts_sec is not None else "normal"
        
        # Save windows
        np.save(output_base_dir / f"{scenario}_{sc_path.stem}_{overall_label}_windows.npy", windows_np)
        # Save per-window labels
        np.save(output_base_dir / f"{scenario}_{sc_path.stem}_{overall_label}_labels.npy", labels_np)

# %%
# 4. Run
print(f"Processing {len(all_test_sc)} recordings with window-level labeling...")
for sc_path in tqdm(all_test_sc, desc="Test-Window-Labeling"):
    process_test_file_with_timestamps(sc_path)

print("\n--- Done! Dataset ready for Phase 3 Rule Extraction ---")
