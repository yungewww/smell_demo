#!/usr/bin/env python3

import os
import torch
import pandas as pd
import numpy as np
from data_loader import highpass_fft_batch
from models.transformer import Transformer
from config import CLASS_LABELS, DEVICE


# ========== Fixed configuration ==========
# CSV_PATH = "data/cloves/cloves.7050bb0ac364.csv"  # cloves
# CSV_PATH = "data/cumin/cumin.dcc4be942e86.csv"  # cumin
# CSV_PATH = "data/oregano.real.552189cd9ba7.csv"  # oregano

CSV_PATH = "data/cumin.today.bd6399a46021.csv"


# CKPT_PATH = (
#     "checkpoints/medium_overlap_win30_str15_prochighpass_fft_batch_20251019_142736.pt"
# )
# WINDOW_SIZE = 30
# STRIDE = 15


CKPT_PATH = (
    "checkpoints/long_overlap_win60_str30_prochighpass_fft_batch_20251019_142823.pt"
)
WINDOW_SIZE = 60
STRIDE = 30
SELECTED_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]  # no benzene
PROCESSING = ["highpass_fft_batch"]


# ========== Utility functions ==========
def load_csv_as_data(path, selected_indices, processing):
    """Read CSV and apply channel selection and preprocessing"""
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number])  # keep numeric columns
    df = df.iloc[:, selected_indices]

    if df.empty:
        raise ValueError("CSV has no numeric data after channel selection.")

    # === Preprocessing ===
    if "highpass_fft_batch" in processing:
        arr = df.to_numpy(dtype=np.float32)[None, :, :]
        arr = highpass_fft_batch(arr)
        df = pd.DataFrame(arr[0], columns=df.columns)

    X = df.to_numpy(dtype=np.float32)
    return X


def sliding_windows(X, window, stride):
    """Split into sliding windows"""
    if len(X) < window:
        return np.expand_dims(X, axis=0)
    windows = np.stack(
        [X[i : i + window] for i in range(0, len(X) - window + 1, stride)],
        axis=0,
    )
    return windows


# ========== Main ==========
if __name__ == "__main__":
    device = DEVICE
    print(f"🖥 Device: {device}")

    # === Load CSV ===
    X = load_csv_as_data(CSV_PATH, SELECTED_INDICES, PROCESSING)
    windows = sliding_windows(X, WINDOW_SIZE, STRIDE)
    print(f"📊 Created {len(windows)} windows from CSV")

    # === Load model ===
    input_dim = windows.shape[2]
    num_classes = len(CLASS_LABELS)

    model = Transformer(
        input_dim=input_dim,
        model_dim=128,
        num_classes=num_classes,
        num_heads=4,
        num_layers=3,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"🧠 Loaded model: {CKPT_PATH}")

    # === Inference ===
    preds = []
    with torch.no_grad():
        for w in windows:
            w = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(w)
            preds.append(logits.argmax(dim=1).item())

    preds = np.array(preds)
    counts = np.bincount(preds, minlength=num_classes)
    winner = counts.argmax()
    class_name = list(CLASS_LABELS.keys())[winner]

    print("\n─────────────── Result ───────────────")
    print(f"📄 CSV file: {CSV_PATH}")
    print(f"🪶 Predicted class: {class_name}")
    print(f"📊 Window votes:")
    for name, idx in CLASS_LABELS.items():
        print(f"{name:<15}: {counts[idx]} windows")
    print("─────────────────────────────────────")