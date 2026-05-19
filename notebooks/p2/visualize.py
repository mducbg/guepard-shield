"""Thesis visualization script for Phase 2.

Generates:
1. Training/Validation Loss curves.
2. ROC and Precision-Recall curves.
3. Anomaly Score Distribution (Bình thường vs Tấn công).
4. Confusion Matrix (at Oracle threshold).

Outputs saved to results/p2/plots/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "results" / "p2" / "checkpoints" / "lightning_logs" / "version_5" / "metrics.csv"
SCORE_DIR = PROJECT_ROOT / "results" / "p2" / "scores"
PLOT_DIR = PROJECT_ROOT / "results" / "p2" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def plot_loss():
    print("Plotting loss curves...")
    df = pd.read_csv(LOG_FILE)
    
    # Columns: epoch,step,train_loss_epoch,train_loss_step,val_loss
    train_df = df[df['train_loss_step'].notna()].copy()
    val_df = df[df['val_loss'].notna()].copy()
    
    plt.figure(figsize=(10, 6))
    if not train_df.empty:
        plt.plot(train_df['step'], train_df['train_loss_step'], label='Train Loss (Step)', alpha=0.3)
        # Smoothing
        window = min(len(train_df), 100)
        plt.plot(train_df['step'], train_df['train_loss_step'].rolling(window=window).mean(), label='Train Loss (Smooth)')
    
    if not val_df.empty:
        plt.plot(val_df['step'], val_df['val_loss'], 'o-', label='Val Loss', linewidth=2)

    plt.xlabel('Global Steps')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training and Validation Loss (Phase 2)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss_curves.png", dpi=300)
    plt.close()

def plot_curves():
    score_path = SCORE_DIR / "test_last.npy"
    label_path = SCORE_DIR / "test_labels.npy"
    
    if not score_path.exists() or not label_path.exists():
        print(f"Warning: Score files not found at {SCORE_DIR}. Skipping performance plots.")
        print("Tip: Update notebooks/p2/eval.py to save scores and run it again.")
        return

    print("Loading scores (this might take a while)...")
    scores = np.load(SCORE_DIR / "test_last.npy")
    labels = np.load(SCORE_DIR / "test_labels.npy")
    
    print("Computing ROC and PR curves...")
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # ROC Plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=300)
    plt.close()
    
    # PR Plot
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pr_curve.png", dpi=300)
    plt.close()
    
    # Distribution Plot (Sampled)
    print("Plotting anomaly score distribution...")
    plt.figure(figsize=(10, 6))
    
    # Sample to avoid memory issues and slow plotting
    sample_size = 100000
    if len(scores) > sample_size:
        indices = np.random.choice(len(scores), sample_size, replace=False)
        s_samp = scores[indices]
        l_samp = labels[indices]
    else:
        s_samp = scores
        l_samp = labels
        
    sns.kdeplot(s_samp[l_samp == 0], fill=True, label='Normal', color='green', alpha=0.5)
    sns.kdeplot(s_samp[l_samp == 1], fill=True, label='Attack', color='red', alpha=0.5)
    plt.axvline(x=2.6055, color='black', linestyle='--', label='Oracle τ (2.606)')
    plt.xlabel('Anomaly Score (Last-token NLL)')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "score_distribution.png", dpi=300)
    plt.close()
    
    # Confusion Matrix at Oracle tau
    print("Computing Confusion Matrix...")
    tau = 2.6055
    preds = (scores >= tau).astype(int)
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (τ = {tau:.4f})')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

def main():
    try:
        plot_loss()
    except Exception as e:
        print(f"Error plotting loss: {e}")
        
    try:
        plot_curves()
    except Exception as e:
        print(f"Error plotting performance curves: {e}")
    
    print(f"\nAll plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
