# %% [markdown]
# # DongTing Dataset — Diagnostic Analysis

# %% Setup
from gp.config import RESULTS_DIR
from gp.data_loader.dongting_loader import DongTingDataset
from gp.diagnostic.stats import Stat
from rich.progress import track

OUT_DIR = RESULTS_DIR / "diagnostic" / "dongting"

dataset = DongTingDataset()
print(f"Dataset contains {len(dataset)} recordings")

# %% Single-pass analysis
stat = Stat()
for recording in track(dataset, total=len(dataset)):
    stat.analyze(recording)

# %% 1. Integrity Check
stat.report_integrity(save_dir=OUT_DIR)

# %% 2. Sequence Length Analysis
stat.report_seq_lengths(save_dir=OUT_DIR)

# %% 3. Syscall Vocabulary Analysis
stat.report_syscall_vocab(save_dir=OUT_DIR)
