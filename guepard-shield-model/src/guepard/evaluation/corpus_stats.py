"""Corpus integrity and window statistics for DongTing-style datasets."""
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from tqdm import tqdm

from ..config import WindowConfig
from ..data_loader.windowing import num_sliding_windows

CountTokens = Optional[Callable[[Path], int]]


def scan_corpus_integrity(
    metadata: list, count_tokens: CountTokens = None
) -> tuple[list[dict], list[tuple]]:
    """Scan all files for missing/empty/mismatched entries.

    Parameters
    ----------
    count_tokens:
        Optional callable ``(file_path: Path) -> int`` that counts the number
        of tokens/events in a file.  When *None* (default) the file is read as
        UTF-8 text and pipe ``"|"`` characters are counted (DongTing format).
        Pass a dataset-specific counter for other formats (e.g. LID-DS .sc
        files use exit-event counting).

    Returns:
        seq_data: list of dicts with keys seq_id, split, label, bug_name,
                  metadata_len, actual_len (None if unreadable), file_path.
        file_issues: list of (seq_id, description) tuples.
    """
    seq_data: list[dict] = []
    file_issues: list[tuple] = []

    for m in tqdm(metadata, desc="Scanning", unit="seq"):
        entry: dict[str, Any] = {
            "seq_id": m.seq_id,
            "split": m.seq_class,
            "label": m.label,
            "bug_name": m.bug_name,
            "metadata_len": m.seq_length,
            "actual_len": None,
            "file_path": m.file_path,
        }
        if not Path(m.file_path).exists():
            file_issues.append((m.seq_id, "missing file"))
            seq_data.append(entry)
            continue
        try:
            if count_tokens is not None:
                actual_len = count_tokens(Path(m.file_path))
                if actual_len == 0:
                    file_issues.append((m.seq_id, "empty file"))
                    seq_data.append(entry)
                    continue
            else:
                content = Path(m.file_path).read_text(encoding="utf-8").strip()
                if not content:
                    file_issues.append((m.seq_id, "empty file"))
                    seq_data.append(entry)
                    continue
                actual_len = content.count("|") + 1
            if actual_len != m.seq_length:
                file_issues.append(
                    (m.seq_id, f"length mismatch: metadata={m.seq_length} actual={actual_len}")
                )
            entry["actual_len"] = actual_len
        except Exception as e:
            file_issues.append((m.seq_id, f"read error: {e}"))
        seq_data.append(entry)

    return seq_data, file_issues


def compute_window_stats(
    valid: list[dict],
    window_config: WindowConfig,
) -> dict[str, dict[int, list[int]]]:
    """Compute per-sequence window counts grouped by split × label.

    Returns:
        win_stats[split][label] = list of window-counts per sequence.
    """
    win_stats: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for s in tqdm(valid, desc="Computing windows", unit="seq"):
        n = num_sliding_windows(s["actual_len"], window_config)
        win_stats[s["split"]][s["label"]].append(n)
    return dict(win_stats)


def recommend_max_windows(win_stats: dict) -> dict:
    """Recommend max_windows_per_seq based on P90 of normal sequences.

    Returns dict with p50_n, p90_n, pct_capped.
    """
    all_normal_wpc: list[int] = []
    all_attack_wpc: list[int] = []
    for by in win_stats.values():
        all_normal_wpc.extend(by.get(0, []))
        all_attack_wpc.extend(by.get(1, []))

    p50_n = int(np.percentile(all_normal_wpc, 50)) if all_normal_wpc else 0
    p90_n = int(np.percentile(all_normal_wpc, 90)) if all_normal_wpc else 0
    pct_capped = (
        float(np.mean(np.array(all_attack_wpc) > p90_n) * 100) if all_attack_wpc else 0.0
    )
    return {"p50_n": p50_n, "p90_n": p90_n, "pct_capped": pct_capped}
