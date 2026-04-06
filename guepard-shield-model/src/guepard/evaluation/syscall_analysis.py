"""Syscall vocab and PMI analysis for DongTing-style datasets."""
from collections import Counter
from typing import Callable, Iterator, Optional

import numpy as np
from tqdm import tqdm

TokenReader = Optional[Callable[[object], list[str]]]
_default_reader = lambda fp: fp.read_text(encoding="utf-8").strip().split("|")


def stream_tokens(
    entries: list[dict], desc: str = "", token_reader: TokenReader = None
) -> Iterator[tuple[dict, list[str]]]:
    """Lazy token stream — reads one file at a time, yields (entry, tokens).

    Parameters
    ----------
    token_reader:
        Optional callable ``(file_path) -> list[str]``.  When *None* (default)
        the file is read as UTF-8 text and split on ``"|"`` (DongTing format).
        Pass a dataset-specific reader for other formats (e.g. LID-DS .sc files).
    """
    reader = token_reader or _default_reader
    for s in tqdm(entries, desc=desc, unit="seq"):
        try:
            tokens = reader(s["file_path"])
            yield s, tokens
        except Exception:
            continue


def count_syscalls(
    valid: list[dict],
    token_reader: TokenReader = None,
) -> tuple[dict[int, Counter], dict[int, Counter]]:
    """Count syscall frequencies and sequence-level occurrences per class.

    Returns:
        syscall_counts[label]: total token counts per class.
        syscall_seq_count[label]: number of sequences containing each syscall.
    """
    syscall_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    syscall_seq_count: dict[int, Counter] = {0: Counter(), 1: Counter()}

    for s, tokens in stream_tokens(valid, desc="Counting syscalls", token_reader=token_reader):
        syscall_counts[s["label"]].update(tokens)
        syscall_seq_count[s["label"]].update(set(tokens))

    return syscall_counts, syscall_seq_count


def syscall_entropy(counter: Counter) -> float:
    """Shannon entropy (bits) of a syscall frequency distribution."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * np.log2(c / total) for c in counter.values() if c > 0))


def compute_pmi(
    syscall_seq_count: dict[int, Counter],
    n_attack_seqs: int,
    n_total_seqs: int,
    shared: set[str],
) -> dict[str, float]:
    """Compute PMI(syscall; attack_class) for all shared syscalls.

    Positive PMI → syscall associated with attack; negative → normal.
    """
    pmi_scores: dict[str, float] = {}
    for syscall in shared:
        p_s = (syscall_seq_count[0][syscall] + syscall_seq_count[1][syscall]) / n_total_seqs
        if p_s == 0:
            continue
        p_s_given_attack = syscall_seq_count[1][syscall] / max(n_attack_seqs, 1)
        if p_s_given_attack > 0:
            pmi_scores[syscall] = float(np.log2(p_s_given_attack / p_s))
    return pmi_scores


def cooccurrence_matrix(
    valid: list[dict], top_syscalls: list[str], token_reader: TokenReader = None
) -> np.ndarray:
    """Build Jaccard-normalised co-occurrence matrix over top_syscalls.

    Returns an (n, n) float array where entry [i,j] = Jaccard similarity.
    """
    n = len(top_syscalls)
    cooc = np.zeros((n, n))
    for _, tokens in stream_tokens(valid, desc="Co-occurrence matrix", token_reader=token_reader):
        token_set = set(tokens)
        for i, si in enumerate(top_syscalls):
            if si not in token_set:
                continue
            for j, sj in enumerate(top_syscalls):
                if sj in token_set:
                    cooc[i, j] += 1

    cooc_norm = np.zeros_like(cooc)
    for i in range(n):
        for j in range(n):
            denom = cooc[i, i] + cooc[j, j] - cooc[i, j]
            cooc_norm[i, j] = cooc[i, j] / denom if denom > 0 else 0.0
    return cooc_norm
