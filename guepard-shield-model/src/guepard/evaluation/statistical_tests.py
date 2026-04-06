"""Statistical tests for sequence-length vs label analysis."""
import numpy as np
from scipy import stats


def length_distribution_tests(
    split_label_lens: dict[str, dict[int, list[int]]],
    splits: list[str],
) -> list[dict]:
    """Run Mann-Whitney U and KS tests for each split (normal vs attack).

    Returns list of dicts with keys: split, mannwhitney, ks (each has stat, p_val, significant).
    """
    results = []
    for split in splits:
        by = split_label_lens.get(split, {})
        n_lens = np.array(by.get(0, []))
        a_lens = np.array(by.get(1, []))
        entry: dict = {"split": split}

        if len(n_lens) >= 2 and len(a_lens) >= 2:
            u_stat, p_mw = stats.mannwhitneyu(n_lens, a_lens, alternative="two-sided")
            entry["mannwhitney"] = {
                "stat": float(u_stat),
                "p_val": float(p_mw),
                "significant": bool(p_mw < 0.05),
                "median_normal": int(np.median(n_lens)),
                "median_attack": int(np.median(a_lens)),
            }
            ks_stat, p_ks = stats.ks_2samp(n_lens, a_lens)
            entry["ks"] = {
                "stat": float(ks_stat),
                "p_val": float(p_ks),
                "significant": bool(p_ks < 0.05),
            }
        else:
            entry["mannwhitney"] = None
            entry["ks"] = None

        results.append(entry)
    return results


def point_biserial(valid: list[dict]) -> tuple[float, float]:
    """Compute point-biserial correlation between seq_length and binary label.

    Returns (r, p_val).
    """
    all_lens = np.array([s["actual_len"] for s in valid])
    all_labels = np.array([s["label"] for s in valid])
    r, p_val = stats.pointbiserialr(all_labels, all_lens)
    return float(r), float(p_val)
