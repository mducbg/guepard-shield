"""Recording and Syscall dataclasses with .sc / .json parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Syscall:
    timestamp: int  # nanoseconds, from .sc column 0
    thread_id: int  # from .sc column 1
    syscall: str    # syscall name, from .sc column 5


@dataclass
class Recording:
    scenario: str
    split: str               # "train" | "val" | "test"
    name: str                # recording directory name
    is_exploit: bool         # from .json "exploit" field
    exploit_times: list[float]  # absolute timestamps in seconds
    warmup_end: float           # absolute timestamp in seconds
    syscalls: list[Syscall]     # eagerly loaded, exit-only


def load_recording(
    sc_path: Path,
    json_path: Path,
    scenario: str,
    split: str,
    name: str,
    is_exploit: bool,
    max_syscalls: int | None = None,
) -> Recording:
    """Parse one recording from its .sc text file and .json metadata.

    Args:
        max_syscalls: Stop after collecting this many exit-syscall events.
            Useful when windowing with a fixed cap — avoids reading entire
            large files. None reads the whole file.
    """
    with open(json_path) as f:
        meta = json.load(f)

    exploit_times = [e["absolute"] for e in meta["time"]["exploit"]]
    warmup_end = meta["time"]["warmup_end"]["absolute"]

    syscalls: list[Syscall] = []
    with open(sc_path) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 7:
                continue
            if fields[6] != "<":
                continue
            syscalls.append(Syscall(
                timestamp=int(fields[0]),
                thread_id=int(fields[1]),
                syscall=fields[5],
            ))
            if max_syscalls is not None and len(syscalls) >= max_syscalls:
                break

    return Recording(
        scenario=scenario,
        split=split,
        name=name,
        is_exploit=is_exploit,
        exploit_times=exploit_times,
        warmup_end=warmup_end,
        syscalls=syscalls,
    )
