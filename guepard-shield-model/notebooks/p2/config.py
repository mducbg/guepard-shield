"""Shared constants for all P2 notebook scripts.

Import this in every p2 notebook:
    from config import DATA_DIR, OUTPUT_DIR, ...
"""
from pathlib import Path

import torch

DATA_DIR = Path("../../data/processed/LID-DS-2021")
OUTPUT_DIR = Path("../../results/p2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_DIST_SCENARIOS = [
    "CVE-2014-0160",        # Heartbleed
    "CVE-2017-7529",        # Nginx OOB read
    "CWE-89-SQL-injection", # SQL Injection
    "Bruteforce_CWE-307",   # Brute force auth
    "EPS_CWE-434",          # Unrestricted file upload
]

SEED = 42
WINDOW_SIZE = 64
STRIDE = 12
MAX_WINDOWS_PER_SEQ = 10
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4
VECTORIZER_MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)
T_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

if torch.cuda.is_available():
    PRECISION = "bf16-mixed" if torch.cuda.get_device_capability()[0] >= 8 else "16-mixed"
else:
    PRECISION = "32-true"

# Artifact paths — written by one notebook, read by the next
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
VECTORIZER_PATH = OUTPUT_DIR / "vectorizer.joblib"
COMPARISON_PATH = OUTPUT_DIR / "teacher_comparison.json"
WINNER_CKPT_PATH = OUTPUT_DIR / "best_teacher_lidds.ckpt"
SWEEP_PATH = OUTPUT_DIR / "temperature_sweep.json"
CHECKPOINT_PATH = OUTPUT_DIR / "p2_checkpoint.json"
