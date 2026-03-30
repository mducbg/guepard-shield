from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..config import WindowConfig


@dataclass
class WindowMeta:
    """Metadata to locate and extract a specific window from a corpus sequence."""

    seq_id: str  # Global ID from Baseline.csv
    label: int  # Ground truth class
    start_idx: int  # Window start position in the sequence
    window_length: int  # Length of this specific window (often config.window_size)
    file_path: Path  # Path to the raw sequence file


def num_sliding_windows(seq_length: int, config: WindowConfig) -> int:
    """Calculates the total number of sliding windows for a given sequence length."""
    if seq_length < config.min_window_size:
        return 0
    if seq_length <= config.window_size:
        return 1

    count = (seq_length - config.window_size) // config.stride + 1

    last_end = (count - 1) * config.stride + config.window_size
    if last_end < seq_length:
        count += 1

    return count


def get_window_meta(
    seq_id: str,
    label: int,
    seq_length: int,
    config: WindowConfig,
    file_path: Path,
    window_idx: int,
) -> WindowMeta:
    """Derives WindowMeta for a sequence without pregenerating the entire array."""
    if seq_length <= config.window_size:
        return WindowMeta(seq_id, label, 0, seq_length, file_path)

    # Calculate base stride displacement
    base_start = window_idx * config.stride
    last_full_window_idx = (seq_length - config.window_size) // config.stride

    # Check if this index belongs to the trailing remainder
    if window_idx > last_full_window_idx:
        start_idx = seq_length - config.window_size
    else:
        start_idx = base_start

    return WindowMeta(seq_id, label, start_idx, config.window_size, file_path)


def extract_window_tokens(tokens: List[str], meta: WindowMeta) -> List[str]:
    """Extracts the slice of tokens specified by WindowMeta."""
    return tokens[meta.start_idx : meta.start_idx + meta.window_length]
