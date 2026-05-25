"""Shared project paths for Phase 1 (EDA & preprocessing).

Phase 2+ scripts (notebooks/p2/) define their own paths inline and do NOT
import from this module.
"""

from __future__ import annotations
from pathlib import Path

# ===========================================================================
# Base Paths
# ===========================================================================

PROJECT_ROOT = Path.cwd() # Root: guepard-shield/
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Extracted data paths
EXTRACTED_DATA_DIR = DATA_ROOT / "extracted"
DONGTING_DIR = EXTRACTED_DATA_DIR / "DongTing"
LIDDS_2019_DIR = EXTRACTED_DATA_DIR / "LID-DS-2019"
LIDDS_2021_DIR = EXTRACTED_DATA_DIR / "LID-DS-2021"

# Processed data paths
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"

# Syscall table
TBL_PATH = DONGTING_DIR / "syscall_64.tbl"

# Common output paths reused by Phase 1 scripts and derived artifacts
npy_dir: Path = PROCESSED_DATA_DIR / "lidds2021"
vocab_path: Path = RESULTS_DIR / "p1" / "vocab_lidds2021_train.txt"
