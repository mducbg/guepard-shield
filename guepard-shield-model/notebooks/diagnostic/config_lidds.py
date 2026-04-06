"""Diagnostic config for LID-DS-2021 dataset."""

from pathlib import Path

from guepard.config import WindowConfig
from guepard.data_loader.lidds_corpus import (
    LiddsCorpus,
    _count_exit_events,
    read_sc_tokens,
)

DATA_DIR = Path("../../data/processed/LID-DS-2021")
OUTPUT_DIR = Path("../../results/diagnostic/lidds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all scenarios for EDA.  To restrict to in-distribution scenarios only:
#   CORPUS = LiddsCorpus(DATA_DIR, scenarios=["CVE-2014-0160", "CVE-2017-7529", ...])
CORPUS = LiddsCorpus(DATA_DIR)
WINDOW_CONFIG = WindowConfig(window_size=64, stride=12)
SPLITS = ["training", "validation", "test"]
TRAIN_SPLIT = "training"
DATASET_NAME = "LID-DS-2021"
CLASS_NAMES = {0: "normal", 1: "attack"}
COLORS = {0: "steelblue", 1: "tomato"}


# LID-DS files are columnar .sc files; exit events are counted via " < " marker
def TOKEN_READER(fp):
    return read_sc_tokens(str(fp))


def COUNT_TOKENS(fp):
    return _count_exit_events(Path(fp))
