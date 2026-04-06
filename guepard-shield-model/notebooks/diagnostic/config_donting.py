"""Diagnostic config for DongTing dataset."""

from pathlib import Path

from guepard.config import WindowConfig
from guepard.data_loader.corpus import DongTingCorpus

DATA_DIR = Path("../../data/processed/DongTing")
OUTPUT_DIR = Path("../../results/diagnostic/donting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS = DongTingCorpus(DATA_DIR)
WINDOW_CONFIG = WindowConfig(window_size=64, stride=12)
SPLITS = ["DTDS-train", "DTDS-validation", "DTDS-test"]
TRAIN_SPLIT = "DTDS-train"
DATASET_NAME = "DongTing"
CLASS_NAMES = {0: "normal", 1: "attack"}
COLORS = {0: "steelblue", 1: "tomato"}

# DongTing files are UTF-8 text, syscalls separated by "|"
TOKEN_READER = None  # use default in stream_tokens
COUNT_TOKENS = None  # use default in scan_corpus_integrity
