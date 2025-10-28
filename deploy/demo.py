import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import joblib

from models.transformer import Transformer
from config import CLASS_LABELS, DEVICE
from data_loader import diff_data_like
from run import format_stimuli


# ===================== CONFIG =====================
with open("config.json", "r") as c:
    config = json.load(c)

DATA_DIR = os.path.join(config["data_dir"], format_stimuli(config))
TARGET_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO"]
CHART_HEIGHT = 560
WINDOW_SIZE = 60
STRIDE = 5

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
MODEL_PATH = "checkpoints/long_overlap_first_4_win60_str5_procstandard_scaler_diff_data_like_20251022_143356.pt"
SCALER_PATH = "checkpoints/long_overlap_first_4_scaler_20251022_143356.pkl"


# ===================== MODEL & SCALER =====================
def load_model():
    input_dim = 4
    num_classes = len(CLASS_LABELS)

    model = Transformer(
        input_dim=input_dim,
        model_dim=128,
        num_classes=num_classes,
        num_heads=4,
        num_layers=3,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Model loaded: {MODEL_PATH}")
    return model


MODEL = load_model()
SCALER = joblib.load(SCALER_PATH)
print(f"✅ Scaler loaded: {SCALER_PATH}")


# ===================== HELPERS =====================
def find_latest_csv():
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    return max(csvs, key=os.path.getmtime) if csvs else None


def empty_timeseries_fig():
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.12)
    for i, col in enumerate(TARGET_COLUMNS):
        r, c = divmod(i, 2)
        r, c = r + 1, c + 1
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines", name=col, line=dict(width=1.5)),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Sample", row=r, col=c)
        fig.update_yaxes(title_text=col, row=r, col=c)
    return fig


def empty_pred_fig():
    fig = go.Figure()
    for cls in CLASS_LABELS.keys():
        fig.add_trace(go.Bar(x=["Prediction"], y=[0], name=cls, showlegend=False))
    fig.update_layout(
        barmode="stack",
        yaxis=dict(range=[0, 1], title="Vote Ratio", tickformat=".0%"),
        xaxis=dict(showticklabels=False),
        height=CHART_HEIGHT,
        title=dict(text="Model Majority Vote", x=0.5),
        showlegend=False,
    )
    return fig


def build_pred_fig_from_votes(vote_counts: dict):
    total = sum(vote_counts.values()) or 1
    best = max(vote_counts, key=vote_counts.get)
    fig = go.Figure()
    for cls, count in vote_counts.items():
        ratio = count / total
        color = "green" if cls == best else "lightgray"
        fig.add_trace(
            go.Bar(
                x=["Prediction"],
                y=[ratio],
                name=cls,
                marker=dict(color=color),
                showlegend=False,
                hovertemplate=f"{cls}: {ratio:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis=dict(range=[0, 1], title="Vote Ratio", tickformat=".0%"),
        xaxis=dict(showticklabels=False),
        height=CHART_HEIGHT,
        title=dict(text=f"Majority Vote — {best}", x=0.5),
    )
    return fig


# def preprocess(df: pd.DataFrame):
<<<<<<< HEAD
=======
#     
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#     df = df.select_dtypes(include=[np.number]).iloc[:, :4].copy()
#     if df.empty or len(df) < 80:
#         print(f"[WARN] CSV invalid: {df.shape}")
#         return None

<<<<<<< HEAD
#     df = diff_data_like(df)

=======

#     df = diff_data_like(df)


>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#     try:
#         df_np = SCALER.transform(df.values)
#     except Exception as e:
#         print(f"[WARN] SCALER transform failed: {e}")
#         return None

#     return df_np


def preprocess(df: pd.DataFrame):
    
    df = df.select_dtypes(include=[np.number]).iloc[:, :4].copy()
    if df.empty or len(df) < 80:
        print(f"[WARN] CSV invalid: {df.shape}")
        return None

<<<<<<< HEAD
    df = df.diff(periods=25).iloc[25:]

=======
    
    df = df.diff(periods=25).iloc[25:]

    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    try:
        df_np = SCALER.transform(df.values)
    except Exception as e:
        print(f"[WARN] SCALER transform failed: {e}")
        return None

    return df_np


# ===================== DASH =====================
app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui"},
    children=[
        html.H3("Live Gas Readings + Model Predictions", style={"marginBottom": "6px"}),
        html.Div(id="file-indicator", style={"marginBottom": "12px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px"},
            children=[
                dcc.Graph(
                    id="timeseries-graph",
                    figure=empty_timeseries_fig(),
                    config={"displayModeBar": False},
                ),
                dcc.Graph(
                    id="pred-graph",
                    figure=empty_pred_fig(),
                    config={"displayModeBar": False},
                ),
            ],
        ),
        dcc.Interval(id="tick-fast", interval=1_000, n_intervals=0),
        dcc.Interval(id="tick-slow", interval=5_000, n_intervals=0),
    ],
)


# ---- FAST: update signals ----
@app.callback(
    Output("timeseries-graph", "figure"),
    Output("file-indicator", "children"),
    Input("tick-fast", "n_intervals"),
)
def update_timeseries(_n):
    latest = find_latest_csv()
    if not latest:
        return empty_timeseries_fig(), "Waiting for CSV..."
    try:
        df = pd.read_csv(latest)
    except Exception as e:
        return empty_timeseries_fig(), f"Read error: {e}"

    tail = df.tail(10000)
    fig = empty_timeseries_fig()
    for i, col in enumerate(TARGET_COLUMNS):
        if col in tail.columns:
            y = pd.to_numeric(tail[col], errors="coerce")
            mask = y.notna()
            fig.data[i].x = list(range(mask.sum()))
            fig.data[i].y = list(y[mask])
    return fig, f"Using: {Path(latest).name}"


# ---- SLOW: update prediction (every 5s) ----
@app.callback(
    Output("pred-graph", "figure"),
    Input("tick-slow", "n_intervals"),
)
def update_predictions(_n):
    latest = find_latest_csv()
    if not latest:
        return empty_pred_fig()

    try:
        df = pd.read_csv(latest)
    except Exception:
        return empty_pred_fig()

    arr = preprocess(df)
    if arr is None or len(arr) < WINDOW_SIZE:
        return empty_pred_fig()

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    windows = [
        arr[i : i + WINDOW_SIZE] for i in range(0, len(arr) - WINDOW_SIZE + 1, STRIDE)
    ]
    preds = []
    with torch.no_grad():
        for w in windows:
            X = torch.tensor(w, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = MODEL(X)
            preds.append(int(torch.argmax(logits, dim=1).cpu()))

<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    idx2cls = {v: k for k, v in CLASS_LABELS.items()}
    counts = {cls: 0 for cls in idx2cls.values()}
    for p in preds:
        counts[idx2cls.get(p, "unknown")] += 1

    return build_pred_fig_from_votes(counts)


if __name__ == "__main__":
    app.run(debug=True)