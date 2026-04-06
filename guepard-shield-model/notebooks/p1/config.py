"""Shared constants for P1 pilot notebooks."""
from pathlib import Path

import torch

DATA_DIR = Path("../../data/processed/DongTing")
OUTPUT_DIR = Path("../../results/pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIMIT = 500
WINDOW_SIZE = 64
STRIDE = 12
MAX_WINDOWS = 5
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 10
TEMPERATURE = 4.0
VECTORIZER_MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)

if torch.cuda.is_available():
    PRECISION = "bf16-mixed" if torch.cuda.get_device_capability()[0] >= 8 else "16-mixed"
else:
    PRECISION = "32-true"

# Artifact paths
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
VECTORIZER_PATH = OUTPUT_DIR / "vectorizer.joblib"
TEACHER_CKPT_PATH = OUTPUT_DIR / "teacher_bilstm.ckpt"
RESULTS_PATH = OUTPUT_DIR / "pilot_results.json"
