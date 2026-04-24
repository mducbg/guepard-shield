"""LID-DS-2019 dataset loader.

Key differences from LID-DS-2021:
- No predefined train/val/test split — all recordings live flat in the scenario dir.
- Metadata comes from a single runs.csv per scenario (not per-recording JSON).
- .txt field layout differs from .sc:
    [0] event_num  [1] HH:MM:SS.ns  [2] cpu  [3] uid  [4] process
    [5] thread_id  [6] direction  [7] syscall_name  [8..] args
- exploit_start_time in runs.csv is seconds-from-recording-start (-1 if no exploit).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gp.data_loader.recording import Syscall

_NON_SCENARIO = {"README.md"}


@dataclass
class LIDDS2019Recording:
    scenario: str
    name: str                  # recording name (stem of .txt file)
    is_exploit: bool
    warmup_time: float         # seconds from recording start
    recording_time: float      # total duration in seconds
    exploit_start_time: float  # seconds from recording start; -1.0 if no exploit
    syscalls: list[Syscall]    # exit-only, eagerly loaded


def _parse_syscalls(path: Path) -> list[Syscall]:
    """Parse a LID-DS-2019 .txt trace, keeping only exit (<) events."""
    syscalls: list[Syscall] = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 8:
                continue
            if fields[6] != "<":
                continue
            syscalls.append(Syscall(
                timestamp=_parse_timestamp_ns(fields[1]),
                thread_id=int(fields[5]),
                syscall=fields[7],
            ))
    return syscalls


def _parse_timestamp_ns(ts: str) -> int:
    """Convert 'HH:MM:SS.nanoseconds' to integer nanoseconds since midnight."""
    time_part, ns_part = ts.rsplit(".", 1)
    h, m, s = time_part.split(":")
    base_ns = (int(h) * 3600 + int(m) * 60 + int(s)) * 1_000_000_000
    return base_ns + int(ns_part.ljust(9, "0")[:9])


def _scenario_dirs(data_dir: Path, scenario: str | None) -> list[Path]:
    if scenario is not None:
        return [data_dir / scenario]
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and p.name not in _NON_SCENARIO
    )


def iter_recordings(
    data_dir: Path,
    scenario: str | None = None,
) -> Iterable[LIDDS2019Recording]:
    """Yield one LIDDS2019Recording at a time."""
    for sc_dir in _scenario_dirs(data_dir, scenario):
        sc_name = sc_dir.name
        runs_csv = sc_dir / "runs.csv"
        if not runs_csv.exists():
            continue

        with open(runs_csv, newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                name = row["scenario_name"].strip()
                txt_path = sc_dir / f"{name}.txt"
                if not txt_path.exists():
                    continue

                yield LIDDS2019Recording(
                    scenario=sc_name,
                    name=name,
                    is_exploit=row["is_executing_exploit"].strip() == "True",
                    warmup_time=float(row["warmup_time"]),
                    recording_time=float(row["recording_time"]),
                    exploit_start_time=float(row["exploit_start_time"]),
                    syscalls=_parse_syscalls(txt_path),
                )


def count_recordings(
    data_dir: Path,
    scenario: str | None = None,
) -> int:
    """Cheap row count from runs.csv files — no .txt I/O."""
    total = 0
    for sc_dir in _scenario_dirs(data_dir, scenario):
        runs_csv = sc_dir / "runs.csv"
        if not runs_csv.exists():
            continue
        with open(runs_csv) as f:
            total += sum(1 for _ in f) - 1  # subtract header
    return total
