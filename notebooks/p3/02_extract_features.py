# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from gp.rules.feature_extractor import WindowFeatureExtractor

# %%
TEST_DIR = Path("data/processed/lidds2021/test")
PSEUDO_LABELS_CSV = Path("results/p3_rule_extraction/pseudo_labels.csv")
VOCAB_PATH = Path("results/eda_cross_dataset/vocab_lidds2021_train.txt")
OUTPUT_DIR = Path("results/p3_rule_extraction")
MAX_WINDOWS_PER_RECORDING = 500
NGRAM_FIT_SIZE = 50_000
BATCH_SIZE = 10_000

# %%
pseudo_df = pd.read_csv(PSEUDO_LABELS_CSV)
# filename column: "Foo_exploit_windows.npy" → match directly in test dir
print(f"Pseudo-labels: {pseudo_df['pseudo_label'].value_counts().to_dict()}")

# %%
extractor = WindowFeatureExtractor(vocab_path=VOCAB_PATH, top_ngrams=100)

# %%
all_windows = []
all_labels = []
all_filenames = []

for _, row in tqdm(pseudo_df.iterrows(), total=len(pseudo_df), desc="Loading windows"):
    win_file = TEST_DIR / row["filename"]
    label_file = TEST_DIR / row["filename"].replace("_windows.npy", "_labels.npy")
    if not win_file.exists() or not label_file.exists():
        continue

    windows = np.load(win_file)
    labels = np.load(label_file)

    if MAX_WINDOWS_PER_RECORDING and len(windows) > MAX_WINDOWS_PER_RECORDING:
        idx = np.random.default_rng(42).choice(len(windows), MAX_WINDOWS_PER_RECORDING, replace=False)
        windows, labels = windows[idx], labels[idx]

    all_windows.append(windows)
    all_labels.append(labels)
    all_filenames.extend([row["filename"]] * len(windows))

X_raw = np.concatenate(all_windows, axis=0)
y_window = np.concatenate(all_labels, axis=0)
print(f"Total windows: {len(X_raw)} | attack={np.sum(y_window==1)} normal={np.sum(y_window==0)}")

# %%
if len(X_raw) > NGRAM_FIT_SIZE:
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(X_raw), NGRAM_FIT_SIZE, replace=False)
    extractor.fit_ngrams(X_raw[fit_idx], y_window[fit_idx])
else:
    extractor.fit_ngrams(X_raw, y_window)

# %%
print(f"Extracting features from {len(X_raw)} windows...")
feats = [extractor.transform(X_raw[i:i+BATCH_SIZE]) for i in tqdm(range(0, len(X_raw), BATCH_SIZE))]
X_features = np.concatenate(feats, axis=0)
feature_names = extractor.get_feature_names()
print(f"Feature matrix: {X_features.shape}")

# %%
out = OUTPUT_DIR / "window_features.npz"
np.savez(
    out,
    X=X_features,
    y=y_window,
    filenames=np.array(all_filenames),
    feature_names=np.array(feature_names),
)
print(f"Saved to: {out}")
