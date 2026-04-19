"""DongTing (DT2022) dataset loader.

Each .log file is a single line of pipe-separated syscall names.
Metadata (split, label, kernel version) comes from Baseline.csv.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# CSV column names
_COL_NAME = "kcb_bug_name"
_COL_SPLIT = "kcb_seq_class"
_COL_LABEL = "kcb_seq_lables"   # note: misspelled in the dataset
_COL_KERNEL = "kcb_master_line_ver"

_SPLIT_MAP = {
    "DTDS-train":      "train",
    "DTDS-validation": "val",
    "DTDS-test":       "test",
}
_LABEL_MAP = {
    "Attach": "abnormal",
    "Normal": "normal",
}


@dataclass
class DongTingRecording:
    name: str            # kcb_bug_name from CSV
    split: str           # "train" | "val" | "test"
    label: str           # "normal" | "abnormal"
    kernel_version: str  # e.g. "5.2", "4.15"
    source: str          # immediate parent directory under Normal_data / Abnormal_data
    syscalls: list[str]  # ordered syscall names (pipe-split, single line)


def _file_index(data_dir: Path) -> dict[str, Path]:
    """Build {filename -> Path} for all .log files under data_dir."""
    return {p.name: p for p in data_dir.rglob("*.log")}


def _log_filename(bug_name: str) -> str:
    """Derive the .log filename from kcb_bug_name.

    Normal entries already carry '.log' suffix; abnormal entries do not.
    All files on disk are prefixed with 'sy_'.
    """
    stem = bug_name.removesuffix(".log")
    return f"sy_{stem}.log"


def iter_recordings(
    data_dir: Path,
) -> Iterable[DongTingRecording]:
    """Yield one DongTingRecording per CSV row, reading the .log file eagerly."""
    index = _file_index(data_dir)

    with open(data_dir / "Baseline.csv", newline="") as f:
        for row in csv.DictReader(f):
            filename = _log_filename(row[_COL_NAME])
            path = index.get(filename)
            if path is None:
                continue  # missing file — skip silently

            raw = path.read_text(encoding="utf-8").strip()
            syscalls = raw.split("|") if raw else []

            yield DongTingRecording(
                name=row[_COL_NAME],
                split=_SPLIT_MAP[row[_COL_SPLIT]],
                label=_LABEL_MAP[row[_COL_LABEL]],
                kernel_version=row[_COL_KERNEL],
                source=path.parent.name,
                syscalls=syscalls,
            )


def count_recordings(data_dir: Path) -> int:
    """Cheap row count from CSV — no .log I/O."""
    with open(data_dir / "Baseline.csv", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))
