import datetime

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from ..data_loader.teacher_dataset import TeacherDataset
from ..data_loader.windowing import extract_window_tokens, get_window_meta
from ..features.vectorizer import SyscallVectorizer
from ..data_loader.lidds_corpus import read_sc_tokens


def extract_val_data(
    model: torch.nn.Module,
    dataset: TeacherDataset,
    vectorizer: SyscallVectorizer,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Extract (val_logits, val_labels, X_tfidf, y_labels) from val dataset.

    val_logits / val_labels come from the neural teacher.
    X_tfidf / y_labels are TF-IDF features + window-level ground-truth for DT training.
    Both use the same windows in the same order (dataset iteration order).
    """
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for token_ids, labels in loader:
            all_logits.append(model(token_ids.to(device)).cpu())
            all_labels.append(labels)

    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)

    all_features, y = [], []
    window_config = dataset.window_config
    for i in range(len(dataset)):
        _, label = dataset[i]
        seq_idx, win_idx = dataset.flat_index[i]
        seq_meta = dataset.sequences[seq_idx]
        wmeta = get_window_meta(
            seq_meta.seq_id, seq_meta.label, seq_meta.seq_length,
            window_config, seq_meta.file_path, win_idx,
        )
        tokens = extract_window_tokens(read_sc_tokens(str(wmeta.file_path)), wmeta)
        all_features.append(vectorizer.transform(tokens))
        y.append(int(label))

    return val_logits, val_labels, np.vstack(all_features), np.array(y)


def platt_scaling(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
    """Find T_calib that minimises NLL on the validation set (Platt scaling)."""
    labels_np = val_labels.numpy().astype(int)

    def nll(T: float) -> float:
        probs = torch.softmax(val_logits / T, dim=-1).numpy()
        return -np.log(probs[np.arange(len(labels_np)), labels_np] + 1e-10).mean()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def _entropy(probs: np.ndarray) -> float:
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1).mean() if len(probs) > 0 else 0.0


def run_temperature_sweep(
    val_logits: torch.Tensor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    T_values: list[float],
    seed: int = 42,
) -> list[dict]:
    """For each T: fit DT on soft labels, measure Attack Fidelity + entropy."""
    results = []
    for T in T_values:
        soft_probs = torch.softmax(val_logits / T, dim=-1).numpy()
        teacher_preds = soft_probs.argmax(axis=1)

        dt = DecisionTreeClassifier(max_depth=5, random_state=seed)
        dt.fit(X_val, (soft_probs[:, 1] > 0.5).astype(int))
        dt_preds = dt.predict(X_val)

        overall_fidelity = float(np.mean(dt_preds == teacher_preds))
        attack_mask = teacher_preds == 1
        attack_fidelity = (
            float(np.mean(dt_preds[attack_mask] == teacher_preds[attack_mask]))
            if attack_mask.sum() > 0 else 0.0
        )

        results.append({
            "T": T,
            "overall_fidelity": overall_fidelity,
            "attack_fidelity": attack_fidelity,
            "attack_entropy": float(_entropy(soft_probs[y_val == 1])),
            "normal_entropy": float(_entropy(soft_probs[y_val == 0])),
            "n_leaves": int(dt.get_n_leaves()),
        })

    return results


def evaluate_p2_checkpoint(
    sweep_results: list[dict],
    T_calib: float,
    T_star: float,
    comparison: dict,
    winner_name: str,
) -> dict:
    """Check three P2 pass/fail criteria and return checkpoint dict."""
    best = next(r for r in sweep_results if r["T"] == T_star)
    fid_at_1 = next(r for r in sweep_results if r["T"] == 1.0)["attack_fidelity"]
    fid_at_3 = next(r for r in sweep_results if r["T"] == 3.0)["attack_fidelity"]

    # C1: attack soft labels must be more uncertain than normal — richer supervision
    # for distillation. ">" not abs() — direction matters per §10 of proposal.
    c1 = best["attack_entropy"] > best["normal_entropy"]
    c2 = T_calib > 1.0
    # C3: soft labels must help — fidelity should increase as T rises from 1→3.
    # Flat curve means surrogate can't learn from soft labels (early warning for RQ2).
    c3 = fid_at_3 > fid_at_1

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "winner": winner_name,
        "winner_val_f1": comparison[winner_name]["best_val_f1"],
        "T_calib": T_calib,
        "T_star": T_star,
        "criteria": {
            "1_attack_entropy_gt_normal_entropy": {
                "pass": c1,
                "attack_entropy": best["attack_entropy"],
                "normal_entropy": best["normal_entropy"],
            },
            "2_T_calib_gt_1": {"pass": c2, "T_calib": T_calib},
            "3_fidelity_increases_T1_to_T3": {
                "pass": c3,
                "fidelity_at_T1": fid_at_1,
                "fidelity_at_T3": fid_at_3,
            },
        },
        "all_pass": c1 and c2 and c3,
    }
