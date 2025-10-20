import os

os.chdir(os.path.dirname(__file__))
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from config import CLASS_LABELS  # ✅ import manual labels from config
from config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE


# ========== Basic Functions ==========
def subtract_first_row(df: pd.DataFrame):
    """Subtract the first row from each column to remove bias."""
    return df - df.iloc[0]


def load_sensor_data(data_path, removed_filtered_columns=None):
    """
    Read all CSVs from data_path/train and data_path/test.
    Each subfolder is treated as a separate class, e.g.:
      data/train/chienan_2_2/*.csv
      data/test/rendai_5_10/*.csv
    """
    if removed_filtered_columns is None:
        removed_filtered_columns = []

    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    # ---------- Training set ----------
    train_dir = os.path.join(data_path, "train")
    if os.path.exists(train_dir):
        for folder_name in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder_name)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        cur_path = os.path.join(folder_path, filename)
                        df = pd.read_csv(cur_path)
                        df = subtract_first_row(df)
                        df = df.drop(columns=removed_filtered_columns, errors="ignore")

                        # ✅ Match class name based on folder name (fuzzy match)
                        label = None
                        for k in CLASS_LABELS.keys():
                            if folder_name.startswith(k):
                                label = k
                                break
                        if label is None:
                            raise ValueError(f"❌ Unmatched class name: {folder_name}")

                        training_data[label].append(df)

    # ---------- Test set ----------
    test_dir = os.path.join(data_path, "test")
    if os.path.exists(test_dir):
        for folder_name in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder_name)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        cur_path = os.path.join(folder_path, filename)
                        df = pd.read_csv(cur_path)
                        df = subtract_first_row(df)
                        df = df.drop(columns=removed_filtered_columns, errors="ignore")

                        # ✅ Match class name as well
                        label = None
                        for k in CLASS_LABELS.keys():
                            if folder_name.startswith(k):
                                label = k
                                break
                        if label is None:
                            raise ValueError(f"❌ Unmatched class name: {folder_name}")

                        testing_data[label].append(df)

    return training_data, testing_data


def build_sliding_data(
    data: dict[str, list[pd.DataFrame]],
    window_size,
    stride,
):
    """Generate sliding window data for each class."""
    X = []
    y = []

    for label_name, dfs in data.items():
        if label_name not in CLASS_LABELS:
            raise ValueError(f"❌ Class not defined in CLASS_LABELS: {label_name}")
        label_id = CLASS_LABELS[label_name]

        for df in dfs:
            for start in range(0, len(df) - window_size + 1, stride):
                window = df.iloc[start : start + window_size].values
                X.append(window)
                y.append(label_id)

    X = np.array(X)  # [N, T, C]
    y = np.array(y)
    return X, y


def diff_data_like(data: dict, periods: int = 25):  # optional
    """Time-series differencing to smooth the signal."""
    out = {}
    for label, dfs in data.items():
        out_list = []
        for df in dfs:
            diff_df = df.diff(periods=periods).iloc[periods:]
            out_list.append(diff_df)
        out[label] = out_list
    return out


def highpass_fft_batch(X, sampling_rate=1.0, cutoff=0.05):  # optional
    """FFT high-pass filter"""
    X = np.asarray(X)
    N, T, C = X.shape
    F = np.fft.rfft(X, axis=1)
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)
    mask = (freqs >= cutoff)[None, :, None]
    F *= mask
    X_clean = np.fft.irfft(F, n=T, axis=1)
    return X_clean


# ========== Main (only print data structure, no training) ==========
if __name__ == "__main__":
    data_path = "data"
    removed_filtered_columns = []

    train_data, test_data = load_sensor_data(data_path, removed_filtered_columns)

    # ---------- TRAIN ----------
    total_train_csv = sum(len(v) for v in train_data.values())
    total_train_windows = 0
    print("[TRAIN DATA]")
    for label, dfs in train_data.items():
        X_label, y_label = build_sliding_data(
            {label: dfs}, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE
        )
        num_windows = len(X_label)
        total_train_windows += num_windows
        print(
            f"• {label} (id={CLASS_LABELS[label]}): {len(dfs)} files, {num_windows} windows, shape=({num_windows}, 100, {X_label.shape[2] if len(X_label)>0 else '?'})"
        )
    print(
        f"Total {len(train_data)} labels, {total_train_csv} CSV files, {total_train_windows} windows"
    )

    # ---------- TEST ----------
    total_test_csv = sum(len(v) for v in test_data.values())
    total_test_windows = 0
    print("[TEST DATA]")
    for label, dfs in test_data.items():
        X_label, y_label = build_sliding_data({label: dfs}, window_size=30, stride=30)
        num_windows = len(X_label)
        total_test_windows += num_windows
        print(
            f"• {label} (id={CLASS_LABELS[label]}): {len(dfs)} files, {num_windows} windows, shape=({num_windows}, 100, {X_label.shape[2] if len(X_label)>0 else '?'})"
        )
    print(
        f"Total {len(test_data)} labels, {total_test_csv} CSV files, {total_test_windows} windows"
    )
