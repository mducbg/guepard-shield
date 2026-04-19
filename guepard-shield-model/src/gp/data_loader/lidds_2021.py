"""LID-DS-2021 dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from gp.data_loader.recording import Recording, load_recording

# Directories that are not scenario folders
_NON_SCENARIO = {"_lidds_seq_lengths.json", "README.md"}


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
) -> Iterable[Recording]:
    """Yield one Recording at a time across all scenarios and splits."""
    for sc_dir in _scenario_dirs(data_dir, scenario):
        sc_name = sc_dir.name

        for split_subdir, split_label, is_exploit in [
            (sc_dir / "training",              "train", False),
            (sc_dir / "validation",            "val",   False),
            (sc_dir / "test" / "normal",       "test",  False),
            (sc_dir / "test" / "normal_and_attack", "test", True),
        ]:
            if not split_subdir.exists():
                continue
            for rec_dir in sorted(split_subdir.iterdir()):
                if not rec_dir.is_dir():
                    continue
                yield load_recording(
                    sc_path=rec_dir / f"{rec_dir.name}.sc",
                    json_path=rec_dir / f"{rec_dir.name}.json",
                    scenario=sc_name,
                    split=split_label,
                    name=rec_dir.name,
                    is_exploit=is_exploit,
                )


def count_recordings(
    data_dir: Path,
    scenario: str | None = None,
) -> int:
    """Cheap directory count — no file I/O."""
    total = 0
    for sc_dir in _scenario_dirs(data_dir, scenario):
        for split_subdir in [
            sc_dir / "training",
            sc_dir / "validation",
            sc_dir / "test" / "normal",
            sc_dir / "test" / "normal_and_attack",
        ]:
            if not split_subdir.exists():
                continue
            total += sum(1 for p in split_subdir.iterdir() if p.is_dir())
    return total
