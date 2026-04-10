# %% [markdown]
# # LID-DS-2021 Dataset — Diagnostic Analysis

# %% Setup
from gp.config import RESULTS_DIR
from gp.data_loader.lidds_2021_loader import LIDDS2021Dataset
from gp.data_pipeline.syscall_table import load_syscall_table
from gp.diagnostic.stats import Stat
from rich.progress import track

OUT_DIR = RESULTS_DIR / "diagnostic" / "lidds_2021"
syscall_table = load_syscall_table()

dataset = LIDDS2021Dataset(shuffle=True, seed=42)
print(f"Dataset contains {len(dataset)} recordings")

# %% Single-pass analysis
stat = Stat(syscall_table=syscall_table)
for recording in track(dataset, total=len(dataset)):
    stat.analyze(recording)

# %% 1. Integrity Check
stat.report_integrity(save_dir=OUT_DIR)

# %% 2. Per-Scenario Class Balance
# → informs class_weight for Teacher training and scenario split strategy for C8
stat.report_scenario_balance(save_dir=OUT_DIR)

# %% 3. Sequence Length Distribution
# → prints recommended max_len (next power-of-2 ≥ p95)
stat.report_seq_lengths(save_dir=OUT_DIR)

# %% 4. Attack Timing
# → validates phase segmenter startup/shutdown_fraction defaults
stat.report_attack_timing(save_dir=OUT_DIR)

# %% 5. Syscall Vocabulary
stat.report_syscall_vocab(save_dir=OUT_DIR)

# %% 6. OOV Rate
# → measures how many syscalls fall outside syscall_64.tbl ([UNK]=335 rate)
stat.report_oov_rate(save_dir=OUT_DIR)
