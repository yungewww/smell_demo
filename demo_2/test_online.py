import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from models.transformer import Transformer
from data_loader import diff_data_like, highpass_fft_batch
from config import CLASS_LABELS, DEVICE

CHECKPOINT_DIR = "checkpoints"
ONLINE_DIR = "data/online"
OUT_DIR = "log"

os.makedirs(OUT_DIR, exist_ok=True)


def load_model(model_path, input_dim, num_classes):
    model = Transformer(
        input_dim=input_dim,
        model_dim=128,
        num_classes=num_classes,
        num_heads=4,
        num_layers=3,
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_online(model, df, window_size, stride, processing_combo):
    
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        return None

    if "diff_data_like" in processing_combo:
        df = df.diff(periods=25).iloc[25:]
    if "highpass_fft_batch" in processing_combo:
        arr = df.to_numpy(dtype=np.float32)[None, :, :]
        arr = highpass_fft_batch(arr)
        df = pd.DataFrame(arr[0], columns=df.columns)

    X = df.to_numpy(dtype=np.float32)
    if len(X) < window_size:
        windows = np.expand_dims(X, axis=0)
    else:
        windows = np.stack(
            [X[i : i + window_size] for i in range(0, len(X) - window_size + 1, stride)]
        )

    preds = []
    with torch.no_grad():
        for w in windows:
            w = torch.tensor(w, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = model(w)
            preds.append(logits.argmax(dim=1).item())

    preds = np.array(preds)
    counts = np.bincount(preds, minlength=len(CLASS_LABELS))
    winner = counts.argmax()
    return list(CLASS_LABELS.keys())[winner], counts


if __name__ == "__main__":
    all_results = []
    model_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    online_files = [f for f in os.listdir(ONLINE_DIR) if f.endswith(".csv")]
    class_names = list(CLASS_LABELS.keys())
    num_classes = len(CLASS_LABELS)

    for model_file in model_files:

        
        parts = model_file.split("_")
        try:
            win = int([p for p in parts if p.startswith("win")][0][3:])
            strd = int([p for p in parts if p.startswith("str")][0][3:])
        except Exception:
            win, strd = 30, 15
        proc = next((p for p in parts if p.startswith("proc")), "")
        proc_list = proc.replace("proc", "").split("_") if proc else []

        model_path = os.path.join(CHECKPOINT_DIR, model_file)
        print(f"🧩 Testing model: {model_file}")


        
        sample_df = pd.read_csv(os.path.join(ONLINE_DIR, online_files[0]))
        input_dim = sample_df.select_dtypes(include=[np.number]).shape[1]
        model = load_model(model_path, input_dim, num_classes)

        for csv_file in online_files:
            path = os.path.join(ONLINE_DIR, csv_file)
            df = pd.read_csv(path)
            pred_class, counts = predict_online(model, df, win, strd, proc_list)
            total = counts.sum()
            acc_frac = counts.max() / total if total else 0

            all_results.append(
                {
                    "Model": model_file,
                    "File": csv_file,
                    "Predicted": pred_class,
                    "Vote_Ratio": round(acc_frac, 3),
                    "Votes": dict(zip(class_names, counts.tolist())),
                }
            )

    out_path = os.path.join(
        OUT_DIR, f"online_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"✅ Saved all results to {out_path}")