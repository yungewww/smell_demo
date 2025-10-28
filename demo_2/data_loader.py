import os

os.chdir(os.path.dirname(__file__))
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from config import CLASS_LABELS
from config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE


def subtract_first_row(df: pd.DataFrame):
    """对 DataFrame 每列减去首行，消除偏置。"""
    return df - df.iloc[0]


def load_sensor_data(data_path, removed_filtered_columns=None):
    """
    从 data_path/train 和 data_path/test 读取所有 CSV。
    支持直接放 CSV 文件（不需要子文件夹）。
    文件名中包含类别关键词（如 bodai、go、rendai 等）。
    """
    if removed_filtered_columns is None:
        removed_filtered_columns = []

    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    train_dir = os.path.join(data_path, "train")
    if os.path.exists(train_dir):
        for filename in os.listdir(train_dir):
            if filename.endswith(".csv"):
                cur_path = os.path.join(train_dir, filename)
                df = pd.read_csv(cur_path)
                if df.empty:
                    print(f"❌ Empty CSV detected: {cur_path}")
                    continue
                df = subtract_first_row(df)
                df = df.drop(columns=removed_filtered_columns, errors="ignore")

                label = None
                for k in CLASS_LABELS.keys():
                    if k in filename:
                        label = k
                        break
                if label is None:
                    raise ValueError(f"❌ 未匹配到已知类别: {filename}")

                training_data[label].append(df)

    test_dir = os.path.join(data_path, "test")
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.endswith(".csv"):
                cur_path = os.path.join(test_dir, filename)
                df = pd.read_csv(cur_path)
                if df.empty:
                    print(f"❌ Empty CSV detected: {cur_path}")
                    continue
                df = subtract_first_row(df)
                df = df.drop(columns=removed_filtered_columns, errors="ignore")

                label = None
                for k in CLASS_LABELS.keys():
                    if k in filename:
                        label = k
                        break
                if label is None:
                    raise ValueError(f"❌ 未匹配到已知类别: {filename}")

                testing_data[label].append(df)

    return training_data, testing_data


def build_sliding_data(data: dict[str, list[pd.DataFrame]], window_size, stride):
    """对每个类生成滑动窗口数据"""
    X = []
    y = []

    for label_name, dfs in data.items():
        if label_name not in CLASS_LABELS:
            raise ValueError(f"❌ 未在 CLASS_LABELS 中定义类别: {label_name}")
        label_id = CLASS_LABELS[label_name]

        for df in dfs:
            for start in range(0, len(df) - window_size + 1, stride):
                window = df.iloc[start : start + window_size].values
                X.append(window)
                y.append(label_id)

    X = np.array(X)
    y = np.array(y)
    return X, y


def diff_data_like(data: dict, periods: int = 25):
    """时间序列差分"""
    out = {}
    for label, dfs in data.items():
        out_list = []
        for df in dfs:
            diff_df = df.diff(periods=periods).iloc[periods:]
            out_list.append(diff_df)
        out[label] = out_list
    return out


def highpass_fft_batch(X, sampling_rate=1.0, cutoff=0.05):
    """FFT 高通滤波"""
    X = np.asarray(X)
    N, T, C = X.shape
    F = np.fft.rfft(X, axis=1)
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)
    mask = (freqs >= cutoff)[None, :, None]
    F *= mask
    X_clean = np.fft.irfft(F, n=T, axis=1)
    return X_clean


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
            f"• {label} (id={CLASS_LABELS[label]}): {len(dfs)} files, {num_windows} windows, shape=({num_windows}, {DEFAULT_WINDOW_SIZE}, {X_label.shape[2] if len(X_label)>0 else '?'})"
        )
    print(
        f"总共 {len(train_data)} 个label, 共 {total_train_csv} 个csv文件，{total_train_windows} 个window"
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
            f"• {label} (id={CLASS_LABELS[label]}): {len(dfs)} files, {num_windows} windows, shape=({num_windows}, 30, {X_label.shape[2] if len(X_label)>0 else '?'})"
        )
    print(
        f"总共 {len(test_data)} 个label, 共 {total_test_csv} 个csv文件，{total_test_windows} 个window"
    )
