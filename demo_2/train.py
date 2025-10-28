# import os
# import sys
# import itertools
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, random_split
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import joblib
# from datetime import datetime
# from sklearn.preprocessing import StandardScaler

# from data_loader import (
#     load_sensor_data,
#     build_sliding_data,
#     diff_data_like,
#     highpass_fft_batch,
# )
# from models.transformer import Transformer
# from config import (
#     WINDOW_CONFIGS,
#     DEVICE,
#     DATA_PATH,
#     SAVE_DIR,
#     EPOCHS,
#     LEARNING_RATE,
#     CLASS_LABELS,
#     SELECTED_INDICES,
# )


# def setup_logger():
#     os.makedirs("log", exist_ok=True)
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_path = os.path.join("log", f"log_{timestamp}.txt")
#     sys.stdout = Logger(log_path)
#     print(f"🧾 Logging to: {log_path}")
#     return log_path


# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.__stdout__
#         self.log = open(filename, "w", encoding="utf-8")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()


# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)


# def evaluate(model, loader, device, num_classes):
#     model.eval()
#     total, correct = 0, 0
#     per_class_total = torch.zeros(num_classes)
#     per_class_correct = torch.zeros(num_classes)
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             preds = model(x).argmax(dim=1)
#             correct_mask = preds == y
#             correct += correct_mask.sum().item()
#             total += y.size(0)
#             for c in range(num_classes):
#                 mask = y == c
#                 per_class_total[c] += mask.sum().item()
#                 per_class_correct[c] += (correct_mask & mask).sum().item()
#     overall_acc = 100 * correct / max(total, 1)
#     per_class_acc = 100 * (per_class_correct / per_class_total.clamp_min(1))
#     return overall_acc, per_class_acc


# def load_online_data(online_dir, window_size, stride):
#     if not os.path.exists(online_dir):
#         return None, []
#     files = [f for f in os.listdir(online_dir) if f.endswith(".csv")]
#     if not files:
#         return None, []
#     all_X, names = [], []
#     for f in files:
#         df = pd.read_csv(os.path.join(online_dir, f))
#         df = df.select_dtypes(include=[np.number])
#         if df.empty:
#             continue
#         name = f.split(".")[0]
#         windows = []
#         for i in range(0, len(df) - window_size + 1, stride):
#             windows.append(df.iloc[i : i + window_size].values)
#         if len(windows) > 0:
#             all_X.append(np.stack(windows))
#             names.append(name)
#     if not all_X:
#         return None, []
#     return np.concatenate(all_X, axis=0), names


# def make_loader(X, y, batch_size=32):
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.long)
#     return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


# def train_and_eval(
#     config_name, window_size, stride, indices, processing_combo, summary
# ):
#     print(f"\n🧩 Config: {config_name} (window={window_size}, stride={stride})")
#     print(f"   Selected Channels: {indices}")
#     print(f"   Data Processing: {processing_combo if processing_combo else 'None'}")

#     train_data, test_data = load_sensor_data(DATA_PATH)
#     if "diff_data_like" in processing_combo:
#         train_data = diff_data_like(train_data)
#         test_data = diff_data_like(test_data)

#     X_train, y_train = build_sliding_data(train_data, window_size, stride)
#     X_test, y_test = build_sliding_data(test_data, window_size, stride)
#     X_train, X_test = X_train[:, :, indices], X_test[:, :, indices]

#     if "highpass_fft_batch" in processing_combo:
#         X_train = highpass_fft_batch(X_train)
#         X_test = highpass_fft_batch(X_test)

#     if "standard_scaler" in processing_combo:
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
#             X_train.shape
#         )
#         X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
#             X_test.shape
#         )

#     num_classes = len(CLASS_LABELS)
#     input_dim = X_train.shape[2]

#     X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
#         y_train, dtype=torch.long
#     )
#     X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
#         y_test, dtype=torch.long
#     )

#     val_ratio = 0.2
#     n_val = int(len(X_train) * val_ratio)
#     n_train = len(X_train) - n_val
#     train_ds, val_ds = random_split(
#         TensorDataset(X_train, y_train),
#         [n_train, n_val],
#         generator=torch.Generator().manual_seed(42),
#     )

#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_ds, batch_size=32)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

#     model = Transformer(
#         input_dim=input_dim,
#         model_dim=128,
#         num_classes=num_classes,
#         num_heads=4,
#         num_layers=3,
#     ).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss()

#     print("🚀 Start Training...\n")

#     patience = 5
#     best_val_acc = 0.0
#     epochs_no_improve = 0
#     best_epoch = 0


#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_loss, total_correct, total_samples = 0.0, 0, 0
#         for x, y in tqdm(
#             train_loader, desc=f"[{config_name}] Epoch {epoch}/{EPOCHS}", leave=False
#         ):
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             logits = model(x)
#             loss = criterion(logits, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * x.size(0)
#             total_correct += (logits.argmax(dim=1) == y).sum().item()
#             total_samples += x.size(0)

#         avg_loss = total_loss / total_samples
#         train_acc = 100 * total_correct / total_samples
#         val_acc, _ = evaluate(model, val_loader, DEVICE, num_classes)
#         print(
#             f"[{config_name}] Epoch {epoch:02d} | loss={avg_loss:.4f} | train={train_acc:.2f}% | val={val_acc:.2f}%"
#         )

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_epoch = epoch
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(
#                     f"⏹ Early stopping triggered after {epoch} epochs "
#                     f"(no improvement for {patience} rounds)."
#                 )
#                 print(
#                     f"🏁 Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
#                 )
#                 break

#     ensure_dir("checkpoints")
#     model_path = os.path.join(
#         "checkpoints",
#         f"{config_name}_win{window_size}_str{stride}_proc{'_'.join(processing_combo) or 'none'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
#     )
#     torch.save(best_state or model.state_dict(), model_path)
#     print(f"💾 Best model saved to: {model_path}")

#     overall_train, per_train = evaluate(model, train_loader, DEVICE, num_classes)
#     overall_test, per_test = evaluate(model, test_loader, DEVICE, num_classes)

#     print("\n📊 Train Results")
#     print(f"Overall Accuracy: {overall_train:.2f}%")
#     for name, i in CLASS_LABELS.items():
#         print(f"{name:<15}: {per_train[i]:.2f}%")

#     print("\n📊 Test Results")
#     print(f"Overall Accuracy: {overall_test:.2f}%")
#     for name, i in CLASS_LABELS.items():
#         print(f"{name:<15}: {per_test[i]:.2f}%")

#     online_dir = "data/online"
#     csv_files = (
#         [f for f in os.listdir(online_dir) if f.endswith(".csv")]
#         if os.path.exists(online_dir)
#         else []
#     )
#     if csv_files:
#         print("\n📊 Online Inference (per-file detailed results)")
#         num_classes = len(CLASS_LABELS)
#         class_names = list(CLASS_LABELS.keys())

#         for csv_file in csv_files:
#             path = os.path.join(online_dir, csv_file)
#             df = pd.read_csv(path).select_dtypes(include=[np.number])
#             df = df.iloc[:, indices]
#             if df.empty:
#                 continue
#             true_label = csv_file.split(".")[0]

#             if "diff_data_like" in processing_combo:
#                 df = df.diff(periods=25).iloc[25:]
#             if "highpass_fft_batch" in processing_combo:
#                 df_np = df.to_numpy(dtype=np.float32)[None, :, :]
#                 df_np = highpass_fft_batch(df_np)
#                 df = pd.DataFrame(df_np[0], columns=df.columns)

#             X = df.to_numpy(dtype=np.float32)
#             if len(X) < window_size:
#                 windows = np.expand_dims(X, axis=0)
#             else:
#                 windows = np.stack(
#                     [
#                         X[i : i + window_size]
#                         for i in range(0, len(X) - window_size + 1, stride)
#                     ]
#                 )

#             if "standard_scaler" in processing_combo:
#                 windows = scaler.transform(
#                     windows.reshape(-1, windows.shape[-1])
#                 ).reshape(windows.shape)

#             preds = []
#             with torch.no_grad():
#                 for w in windows:
#                     w = torch.tensor(w, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#                     logits = model(w)
#                     preds.append(logits.argmax(dim=1).item())
#             preds = np.array(preds)
#             counts = np.bincount(preds, minlength=num_classes)
#             winner = counts.argmax()
#             pred_class = class_names[winner]
#             correct = "✅" if pred_class == true_label else "❌"

#             print("──────────────────────────────────────────────")
#             print(f"📄 File: {csv_file}")
#             print(f"🎯 True label: {true_label}")
#             print(f"🪶 Predicted: {pred_class} {correct}")
#             print(f"📊 Window votes:")
#             for name, idx in CLASS_LABELS.items():
#                 print(f"{name:<15}: {counts[idx]} windows")
#             print(f"🏁 Final Decision: {pred_class} ({counts[winner]} / {len(preds)})")
#             print("──────────────────────────────────────────────")

#     summary.append(
#         {
#             "config": config_name,
#             "channels": str(indices),
#             "processing": "_".join(processing_combo) if processing_combo else "none",
#             "window": window_size,
#             "stride": stride,
#             "train_acc": overall_train,
#             "test_acc": overall_test,
#         }
#     )


# if __name__ == "__main__":
#     log_path = setup_logger()
#     print("📦 Multi-Condition Training + Eval (No Save) Start\n")

#     summary = []
#     selected_sets = list(SELECTED_INDICES.values())
#     processing_keys = ["standard_scaler", "diff_data_like", "highpass_fft_batch"]
#     processing_combos = [()]
#     for i in range(1, len(processing_keys) + 1):
#         processing_combos += list(itertools.combinations(processing_keys, i))

#     for config_name, cfg in WINDOW_CONFIGS.items():
#         for indices in selected_sets:
#             for proc_combo in processing_combos:
#                 train_and_eval(
#                     config_name,
#                     cfg["window_size"],
#                     cfg["stride"],
#                     indices,
#                     list(proc_combo),
#                     summary,
#                 )

#     print("\n\n📈 Summary of All Configs")
#     print(
#         "────────────────────────────────────────────────────────────────────────────"
#     )
#     print(
#         f"{'Config':<15} {'Channels':<22} {'Processing':<25} {'Win':<6} {'Str':<6} {'Train(%)':<10} {'Test(%)':<10}"
#     )
#     print("-" * 100)
#     for r in summary:
#         print(
#             f"{r['config']:<15} {r['channels']:<22} {r['processing']:<25} "
#             f"{r['window']:<6} {r['stride']:<6} {r['train_acc']:<10.2f} {r['test_acc']:<10.2f}"
#         )
#     print(
#         "────────────────────────────────────────────────────────────────────────────"
#     )
#     print("🎉 All training + evaluation finished.")
#     print(f"📁 Logs saved to: {log_path}")
import os
import sys
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from data_loader import (
    load_sensor_data,
    build_sliding_data,
    diff_data_like,
    highpass_fft_batch,
)
from models.transformer import Transformer
from config import (
    WINDOW_CONFIGS,
    DEVICE,
    DATA_PATH,
    SAVE_DIR,
    EPOCHS,
    LEARNING_RATE,
    CLASS_LABELS,
    SELECTED_INDICES,
)


def setup_logger():
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("log", f"log_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    print(f"🧾 Logging to: {log_path}")
    return log_path


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    per_class_total = torch.zeros(num_classes)
    per_class_correct = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct_mask = preds == y
            correct += correct_mask.sum().item()
            total += y.size(0)
            for c in range(num_classes):
                mask = y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (correct_mask & mask).sum().item()
    overall_acc = 100 * correct / max(total, 1)
    per_class_acc = 100 * (per_class_correct / per_class_total.clamp_min(1))
    return overall_acc, per_class_acc


def load_online_data(online_dir, window_size, stride):
    if not os.path.exists(online_dir):
        return None, []
    files = [f for f in os.listdir(online_dir) if f.endswith(".csv")]
    if not files:
        return None, []
    all_X, names = [], []
    for f in files:
        df = pd.read_csv(os.path.join(online_dir, f))
        df = df.select_dtypes(include=[np.number])
        if df.empty:
            continue
        name = f.split(".")[0]
        windows = []
        for i in range(0, len(df) - window_size + 1, stride):
            windows.append(df.iloc[i : i + window_size].values)
        if len(windows) > 0:
            all_X.append(np.stack(windows))
            names.append(name)
    if not all_X:
        return None, []
    return np.concatenate(all_X, axis=0), names


def make_loader(X, y, batch_size=32):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def train_and_eval(
    config_name, window_size, stride, indices, processing_combo, summary
):
    print(f"\n🧩 Config: {config_name} (window={window_size}, stride={stride})")
    print(f"   Selected Channels: {indices}")
    print(f"   Data Processing: {processing_combo if processing_combo else 'None'}")

    train_data, _ = load_sensor_data(DATA_PATH)
    if "diff_data_like" in processing_combo:
        train_data = diff_data_like(train_data)

    X_train, y_train = build_sliding_data(train_data, window_size, stride)
    X_train = X_train[:, :, indices]

    if "highpass_fft_batch" in processing_combo:
        X_train = highpass_fft_batch(X_train)

    if "standard_scaler" in processing_combo:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
            X_train.shape
        )

    num_classes = len(CLASS_LABELS)
    input_dim = X_train.shape[2]

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.long
    )

    val_ratio = 0.2
    n_val = int(len(X_train) * val_ratio)
    n_train = len(X_train) - n_val
    train_ds, val_ds = random_split(
        TensorDataset(X_train, y_train),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = Transformer(
        input_dim=input_dim,
        model_dim=128,
        num_classes=num_classes,
        num_heads=4,
        num_layers=3,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Start Training...\n")

    patience = 5
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in tqdm(
            train_loader, desc=f"[{config_name}] Epoch {epoch}/{EPOCHS}", leave=False
        ):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        train_acc = 100 * total_correct / total_samples
        val_acc, _ = evaluate(model, val_loader, DEVICE, num_classes)
        print(
            f"[{config_name}] Epoch {epoch:02d} | loss={avg_loss:.4f} | train={train_acc:.2f}% | val={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"⏹ Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {patience} rounds)."
                )
                print(
                    f"🏁 Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
                )
                break

    ensure_dir("checkpoints")
    model_path = os.path.join(
        "checkpoints",
        f"{config_name}_win{window_size}_str{stride}_proc{'_'.join(processing_combo) or 'none'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
    )
    torch.save(best_state or model.state_dict(), model_path)
    print(f"💾 Best model saved to: {model_path}")

    overall_train, per_train = evaluate(model, train_loader, DEVICE, num_classes)

    print("\n📊 Train Results")
    print(f"Overall Accuracy: {overall_train:.2f}%")
    for name, i in CLASS_LABELS.items():
        print(f"{name:<15}: {per_train[i]:.2f}%")

    online_dir = "data/online"
    csv_files = (
        [f for f in os.listdir(online_dir) if f.endswith(".csv")]
        if os.path.exists(online_dir)
        else []
    )

    if csv_files:
        print("\n📊 Online Inference (per-file detailed results)")
        num_classes = len(CLASS_LABELS)
        class_names = list(CLASS_LABELS.keys())

        total_windows = 0
        correct_windows = 0

        for csv_file in csv_files:
            path = os.path.join(online_dir, csv_file)
            df = pd.read_csv(path).select_dtypes(include=[np.number])
            df = df.iloc[:, indices]
            if df.empty:
                continue
            true_label = csv_file.split(".")[0]

            if "diff_data_like" in processing_combo:
                df = df.diff(periods=25).iloc[25:]
            if "highpass_fft_batch" in processing_combo:
                df_np = df.to_numpy(dtype=np.float32)[None, :, :]
                df_np = highpass_fft_batch(df_np)
                df = pd.DataFrame(df_np[0], columns=df.columns)

            X = df.to_numpy(dtype=np.float32)
            if len(X) < window_size:
                windows = np.expand_dims(X, axis=0)
            else:
                windows = np.stack(
                    [
                        X[i : i + window_size]
                        for i in range(0, len(X) - window_size + 1, stride)
                    ]
                )

            if "standard_scaler" in processing_combo:
                windows = scaler.transform(
                    windows.reshape(-1, windows.shape[-1])
                ).reshape(windows.shape)

            preds = []
            with torch.no_grad():
                for w in windows:
                    w = torch.tensor(w, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    logits = model(w)
                    preds.append(logits.argmax(dim=1).item())
            preds = np.array(preds)
            counts = np.bincount(preds, minlength=num_classes)
            winner = counts.argmax()
            pred_class = class_names[winner]
            correct = "✅" if pred_class == true_label else "❌"

            true_idx = CLASS_LABELS.get(true_label, -1)
            if true_idx != -1:
                correct_windows += (preds == true_idx).sum()
            total_windows += len(preds)

            print("──────────────────────────────────────────────")
            print(f"📄 File: {csv_file}")
            print(f"🎯 True label: {true_label}")
            print(f"🪶 Predicted: {pred_class} {correct}")
            print(f"📊 Window votes:")
            for name, idx in CLASS_LABELS.items():
                print(f"{name:<15}: {counts[idx]} windows")
            print(f"🏁 Final Decision: {pred_class} ({counts[winner]} / {len(preds)})")
            print("──────────────────────────────────────────────")

        if total_windows > 0:
            final_window_acc = correct_windows / total_windows
            print(f"\n✅ Final window-level accuracy: {final_window_acc:.3f}")

    summary.append(
        {
            "config": config_name,
            "channels": str(indices),
            "processing": "_".join(processing_combo) if processing_combo else "none",
            "window": window_size,
            "stride": stride,
            "train_acc": overall_train,
        }
    )


if __name__ == "__main__":
    log_path = setup_logger()
    print("📦 Multi-Condition Training + Eval (Train + Online Only)\n")

    summary = []
    selected_sets = list(SELECTED_INDICES.values())
    processing_keys = ["standard_scaler", "diff_data_like", "highpass_fft_batch"]
    processing_combos = [()]
    for i in range(1, len(processing_keys) + 1):
        processing_combos += list(itertools.combinations(processing_keys, i))

    for config_name, cfg in WINDOW_CONFIGS.items():
        # for indices in selected_sets:
        for set_name, indices in SELECTED_INDICES.items():
            for proc_combo in processing_combos:
                train_and_eval(
                    f"{config_name}_{set_name}",
                    cfg["window_size"],
                    cfg["stride"],
                    indices,
                    list(proc_combo),
                    summary,
                )

    print("\n\n📈 Summary of All Configs")
    print(
        "────────────────────────────────────────────────────────────────────────────"
    )
    print(
        f"{'Config':<15} {'Channels':<22} {'Processing':<25} {'Win':<6} {'Str':<6} {'Train(%)':<10}"
    )
    print("-" * 90)
    for r in summary:
        print(
            f"{r['config']:<15} {r['channels']:<22} {r['processing']:<25} "
            f"{r['window']:<6} {r['stride']:<6} {r['train_acc']:<10.2f}"
        )
    print(
        "────────────────────────────────────────────────────────────────────────────"
    )
    print("🎉 All training + online evaluation finished.")
    print(f"📁 Logs saved to: {log_path}")
