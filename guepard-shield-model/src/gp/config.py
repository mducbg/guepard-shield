"""Configuration module for guepard-shield-model.

Defines data paths and constants used across the codebase.
"""

from pathlib import Path

# Base paths
# The data directory is inside guepard-shield-model/, not at project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_ROOT = Path(__file__).parent.parent.parent  # guepard-shield-model/
DATA_ROOT = MODEL_ROOT / "data"

# Raw data paths
RAW_DATA_DIR = DATA_ROOT / "raw"
DONGTING_ZIP = RAW_DATA_DIR / "DongTing.zip"
LIDDS_2021_ZIP = RAW_DATA_DIR / "LID-DS-2021.zip"

# Extracted data paths
EXTRACTED_DATA_DIR = DATA_ROOT / "extracted"
DONGTING_DIR = EXTRACTED_DATA_DIR / "DongTing"
LIDDS_2019_DIR = EXTRACTED_DATA_DIR / "LID-DS-2019"
LIDDS_2021_DIR = EXTRACTED_DATA_DIR / "LID-DS-2021"

# Processed data paths (ArrayRecord files, gitignored)
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

# Results
RESULTS_DIR = MODEL_ROOT / "results"

# Splits directory
SPLITS_DIR = DATA_ROOT / "splits"

# Syscall table
TBL_PATH = DONGTING_DIR / "syscall_64.tbl"
