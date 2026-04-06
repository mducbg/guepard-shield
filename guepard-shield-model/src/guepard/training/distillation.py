"""Knowledge distillation utilities: extract soft labels from a teacher model."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data_loader.surrogate_dataset import SurrogateDataset
from ..data_loader.teacher_dataset import TeacherDataset


def extract_features_and_soft_labels(
    surrogate_dataset: SurrogateDataset,
    teacher_dataset: TeacherDataset,
    model: nn.Module,
    temperature: float,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass extraction: TF-IDF features, hard labels, and soft labels.

    Returns:
        X: (N, F) float32 feature matrix.
        y_hard: (N,) int array of ground-truth labels.
        y_soft: (N, C) float32 soft probability matrix at given temperature.
    """
    assert len(surrogate_dataset) == len(teacher_dataset), (
        f"Dataset length mismatch: surrogate={len(surrogate_dataset)} teacher={len(teacher_dataset)}"
    )
    device = next(model.parameters()).device
    surrogate_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=False)
    teacher_loader = DataLoader(teacher_dataset, batch_size=batch_size, shuffle=False)

    f_list, h_list, logits_list = [], [], []
    model.eval()
    with torch.no_grad():
        for (feats, hard_y), (token_ids, _) in zip(surrogate_loader, teacher_loader):
            f_list.append(feats.numpy())
            h_list.append(hard_y.numpy())
            logits_list.append(model(token_ids.to(device)).cpu())

    logits = torch.cat(logits_list, dim=0)
    y_soft = F.softmax(logits / temperature, dim=-1).numpy()
    return np.vstack(f_list), np.concatenate(h_list), y_soft
