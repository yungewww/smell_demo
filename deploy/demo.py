# # # demo_interface.py
# # import os
# # import glob
# # import json
# # from pathlib import Path
# # from datetime import datetime

# # import numpy as np
# # import pandas as pd
# # from dash import Dash, dcc, html, Output, Input, no_update
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots
# # from run import format_stimuli  # your helper

# # import torch
# # from models.transformer import Transformer
# # from config import CLASS_LABELS, DEVICE

# # # ===================== CONFIG =====================
# # with open("config.json", "r") as c:
# #     config = json.load(c)

# # DATA_DIR = os.path.join(config["data_dir"], format_stimuli(config))
# # TARGET_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO"]  # 2x2 grid
# # WINDOW_SIZE = 10_000  # tail shown in plots
# # PRED_CLASSES = ["oregano", "cloves", "cumin"]  # ambient removed
# # MODEL_WINDOW = None  # or set to an int if model wants a tail window
# # CHART_HEIGHT = 560  # both panels match height


# # # ---- Start time ----
# # def parse_start_time(s: str):
# #     if not s:
# #         return None, "N/A"
# #     try:
# #         ts = pd.to_datetime(s, utc=False, errors="coerce")
# #         if pd.isna(ts):
# #             raise ValueError("Could not parse date")
# #         dt = ts.to_pydatetime()
# #         return dt, dt.strftime("%Y-%m-%d %H:%M:%S")
# #     except Exception:
# #         return None, f"Unparsed: {s}"


# # START_DT, START_DT_PRETTY = parse_start_time(config.get("start_time", None))


# # def seconds_since(start_dt: datetime) -> int:
# #     if start_dt is None:
# #         return 0
# #     now = (
# #         datetime.now(start_dt.tzinfo)
# #         if getattr(start_dt, "tzinfo", None)
# #         else datetime.now()
# #     )
# #     return max(int((now - start_dt).total_seconds()), 0)


# # def format_hms(total_seconds: int) -> str:
# #     h = total_seconds // 3600
# #     m = (total_seconds % 3600) // 60
# #     s = total_seconds % 60
# #     return f"{h:d}:{m:02d}:{s:02d}"


# # # ---- Duration (flat at config root) ----
# # def standardize_time_seconds(cfg):
# #     raw = cfg.get("time", 0)
# #     try:
# #         val = float(str(raw).strip())
# #     except Exception:
# #         raise ValueError(f"Invalid time value: {raw!r}")
# #     if val <= 0:
# #         raise ValueError("time must be > 0")
# #     unit = (cfg.get("time_units") or "seconds").strip().lower()
# #     multipliers = {
# #         "seconds": 1,
# #         "second": 1,
# #         "s": 1,
# #         "minutes": 60,
# #         "minute": 60,
# #         "m": 60,
# #         "milliseconds": 1e-3,
# #         "millisecond": 1e-3,
# #         "ms": 1e-3,
# #     }
# #     if unit not in multipliers:
# #         raise ValueError(f"Unsupported time unit: {unit}")
# #     seconds = val * multipliers[unit]
# #     cfg["time"], cfg["time_units"] = str(seconds), "seconds"  # normalize
# #     return seconds


# # try:
# #     DURATION_SECONDS = standardize_time_seconds(config)  # e.g., "10" + "minutes" works
# # except Exception:
# #     DURATION_SECONDS = None  # run indefinitely if malformed/missing

# # # ===================== MODEL HOOKS =====================
# # # def load_model():
# # #     # Replace with your real model loader (joblib / torch / TF)
# # #     return None


# # def load_model():
<<<<<<< HEAD
# #     model_path = "checkpoints/long_overlap_first_4_win60_str5_procstandard_scaler_diff_data_like_20251022_134936.pt"

# #     num_classes = len(CLASS_LABELS)

=======

# #     model_path = "checkpoints/long_overlap_first_4_win60_str5_procstandard_scaler_diff_data_like_20251022_134936.pt"


# #     num_classes = len(CLASS_LABELS)


>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
# #     model = Transformer(
# #         input_dim=input_dim,
# #         model_dim=128,
# #         num_classes=num_classes,
# #         num_heads=4,
# #         num_layers=3,
# #     ).to(DEVICE)

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
# #     state = torch.load(model_path, map_location=DEVICE)
# #     model.load_state_dict(state)
# #     model.eval()

# #     print(f"✅ Model loaded: {model_path}")
# #     return model


# # # def predict_proba_from_dataframe(model, df: pd.DataFrame, classes):
<<<<<<< HEAD
# # #     """
# # #     Return dict {class_name: probability} summing to ~1.
# # #     Replace with your real model inference.
# # #     """
# # #     # ----- DUMMY: delete/replace -----
# # #     n = len(classes)
# # #     if len(df) == 0:
# # #         base = np.ones(n) / n
# # #     else:
# # #         base = np.ones(n) / n
# # #         base[len(df) % n] += 0.25
# # #     base = np.clip(base, 1e-9, None)
# # #     base = base / base.sum()
# # #     return {c: float(p) for c, p in zip(classes, base)}
# # #     # ---------------------------------


# # def predict_proba_from_dataframe(model, df: pd.DataFrame, classes):
# #     """
# #     """
=======
# # #     
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
# #     import torch
# #     import numpy as np

# #     if model is None or len(df) == 0:
# #         return {c: 1 / len(classes) for c in classes}

# #     df = df.select_dtypes(include=[np.number]).iloc[:, :4]

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
# #     X = torch.tensor(df.select_dtypes(include=[np.number]).values, dtype=torch.float32)
# #     if X.ndim == 2:
# #         X = X.unsqueeze(0)  # (1, T, D)

# #     with torch.no_grad():
# #         logits = model(X.to(DEVICE))
# #         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

# #     return {cls: float(p) for cls, p in zip(classes, probs)}


# # MODEL = load_model()


# # # ===================== HELPERS =====================
# # def find_latest_csv():
# #     csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
# #     return max(csvs, key=os.path.getmtime) if csvs else None


# # def empty_timeseries_fig():
# #     fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.12)
# #     for i, col in enumerate(TARGET_COLUMNS):
# #         r, c = divmod(i, 2)
# #         r, c = r + 1, c + 1
# #         fig.add_trace(
# #             go.Scatter(x=[], y=[], mode="lines", name=col, line=dict(width=1.5)),
# #             row=r,
# #             col=c,
# #         )
# #         fig.update_xaxes(title_text="Sample", row=r, col=c)
# #         fig.update_yaxes(title_text=col, row=r, col=c)
# #     # fig.update_layout(height=CHART_HEIGHT, title_text="Live Gas Readings", showlegend=False)
# #     return fig


# # def empty_pred_fig():
# #     fig = go.Figure()
# #     for cls in PRED_CLASSES:
# #         fig.add_trace(
# #             go.Bar(
# #                 x=["Prediction"],
# #                 y=[0],
# #                 name=cls,
# #                 marker=dict(color="lightgray"),
# #                 showlegend=False,
# #             )
# #         )
# #     fig.update_layout(
# #         barmode="stack",
# #         yaxis=dict(range=[0, 1], title="Probability", tickformat=".0%"),
# #         xaxis=dict(showticklabels=False),
# #         height=CHART_HEIGHT,
# #         margin=dict(l=60, r=40, t=70, b=40),
# #         title=dict(text="Model Predictions (Whole CSV)", x=0.5, y=0.98),
# #         showlegend=False,
# #     )
# #     return fig


# # def build_pred_fig_from_probs(vals: dict):
# #     

<<<<<<< HEAD
# #     fig.update_layout(
# #         barmode="stack",
# #         yaxis=dict(range=[0, 1], title="Probability", tickformat=".0%"),
# #         xaxis=dict(showticklabels=False),
# #         height=CHART_HEIGHT,
# #         margin=dict(l=60, r=40, t=70, b=40),
# #         title=dict(text="Model Predictions (Whole CSV)", x=0.5, y=0.98),
# #         showlegend=False,
# #     )
# #     return fig


# # # ===================== DASH APP =====================
# # app = Dash(__name__)
# # app.layout = html.Div(
# #     style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui"},
# #     children=[
# #         dcc.Location(id="url"),  # <-- triggers predictions on page load/refresh
# #         html.H3("Live Gas Readings + Model Predictions", style={"marginBottom": "6px"}),
# #         html.Div(id="file-indicator", style={"opacity": 0.85, "marginBottom": "12px"}),
# #         html.Div(
# #             style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px"},
# #             children=[
# #                 dcc.Graph(
# #                     id="timeseries-graph",
# #                     figure=empty_timeseries_fig(),
# #                     config={"displayModeBar": False},
# #                 ),
# #                 dcc.Graph(
# #                     id="pred-graph",
# #                     figure=empty_pred_fig(),
# #                     config={"displayModeBar": False},
# #                 ),
# #             ],
# #         ),
# #         # Two intervals: fast (1s) for sensors/meta; slow (10s) for model predictions
# #         dcc.Interval(id="tick-fast", interval=1_000, n_intervals=0, disabled=False),
# #         dcc.Interval(id="tick-slow", interval=10_000, n_intervals=0, disabled=False),
# #     ],
# # )


# # # ---- FAST callback: updates sensor figure + status line; also disables both when duration is over
# # @app.callback(
# #     Output("timeseries-graph", "figure"),
# #     Output("file-indicator", "children"),
# #     Output("tick-fast", "disabled"),
# #     Output("tick-slow", "disabled"),
# #     Input("tick-fast", "n_intervals"),
# # )
# # def update_timeseries(_n_fast):
# #     # compute elapsed / remaining and whether we're done
# #     elapsed_sec = seconds_since(START_DT)
# #     if DURATION_SECONDS:
# #         done = elapsed_sec >= DURATION_SECONDS
# #         elapsed_capped = min(elapsed_sec, int(DURATION_SECONDS))
# #         remaining = max(int(DURATION_SECONDS) - elapsed_sec, 0)
# #     else:
# #         done = False
# #         elapsed_capped = elapsed_sec
# #         remaining = None

# #     elapsed_str = format_hms(elapsed_capped)
# #     remaining_str = format_hms(remaining) if remaining is not None else "N/A"
# #     start_pretty = START_DT_PRETTY

# #     latest = find_latest_csv()
# #     if not latest:
# #         meta = f"Start: {start_pretty} • Elapsed: {elapsed_str} • Remaining: {remaining_str} • Waiting for file..."
# #         return empty_timeseries_fig(), meta, done, done

# #     # read CSV and update 2x2 figure
# #     try:
# #         df_full = pd.read_csv(latest)
# #     except Exception as e:
# #         meta = f"Start: {start_pretty} • Elapsed: {elapsed_str} • Remaining: {remaining_str} • Read error: {e}"
# #         return empty_timeseries_fig(), meta, done, done

# #     ts_fig = empty_timeseries_fig()
# #     tail = df_full.tail(WINDOW_SIZE)
# #     for i, col in enumerate(TARGET_COLUMNS):
# #         if col in tail.columns:
# #             y = pd.to_numeric(tail[col], errors="coerce")
# #             mask = y.notna()
# #             ts_fig.data[i].x = list(range(mask.sum()))
# #             ts_fig.data[i].y = list(y[mask])
# #     # ts_fig.update_layout(title_text=f"Live Gas Readings — {Path(latest).name}", height=CHART_HEIGHT)

# #     meta = f"Start: {start_pretty} • Elapsed: {elapsed_str} • Remaining: {remaining_str} • Using: {Path(latest).name}"
# #     return ts_fig, meta, done, done


# # # ---- SLOW callback: updates predictions every 10s AND on page load/refresh
# # @app.callback(
# #     Output("pred-graph", "figure"),
# #     Input("tick-slow", "n_intervals"),
# #     Input("url", "href"),  # <-- triggers once on page load/refresh
# # )
# # def update_predictions(_n_slow, _href):
# #     # Even if the duration has passed, we still evaluate once whenever the page loads/refreshes.
# #     latest = find_latest_csv()
# #     if not latest:
# #         return empty_pred_fig()

# #     try:
# #         df_full = pd.read_csv(latest)
# #     except Exception:
# #         return empty_pred_fig()

# #     df_for_model = df_full if MODEL_WINDOW is None else df_full.tail(MODEL_WINDOW)
# #     probs = predict_proba_from_dataframe(MODEL, df_for_model, PRED_CLASSES)

# #     # normalize
# #     vals = {k: max(0.0, float(v)) for k, v in probs.items() if k in PRED_CLASSES}
# #     s = sum(vals.values()) or 1.0
# #     vals = {k: v / s for k, v in vals.items()}

# #     return build_pred_fig_from_probs(vals)


# # if __name__ == "__main__":
# #     # Dash 3+: use run()
# #     app.run(debug=True)

# # demo.py
# import os
# import glob
# import json
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import pandas as pd
# from dash import Dash, dcc, html, Output, Input
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# import torch
# from models.transformer import Transformer
# from config import CLASS_LABELS, DEVICE
# from data_loader import diff_data_like, highpass_fft_batch
# from sklearn.preprocessing import StandardScaler
# from run import format_stimuli

# # ===================== CONFIG =====================
# with open("config.json", "r") as c:
#     config = json.load(c)

# DATA_DIR = os.path.join(config["data_dir"], format_stimuli(config))
# TARGET_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO"]
# PRED_CLASSES = ["oregano", "cloves", "cumin"]
# CHART_HEIGHT = 560
# WINDOW_SIZE = 60
# STRIDE = 5


# # ===================== MODEL =====================
# def load_model():
#     model_path = "checkpoints/long_overlap_first_4_win60_str5_procstandard_scaler_diff_data_like_20251022_134936.pt"
#     input_dim = 4
#     num_classes = len(CLASS_LABELS)

#     model = Transformer(
#         input_dim=input_dim,
#         model_dim=128,
#         num_classes=num_classes,
#         num_heads=4,
#         num_layers=3,
#     ).to(DEVICE)

#     state = torch.load(model_path, map_location=DEVICE)
#     model.load_state_dict(state)
#     model.eval()
#     print(f"✅ Model loaded: {model_path}")
#     return model


# MODEL = load_model()
# SCALER = StandardScaler()


# # ===================== HELPERS =====================
# def find_latest_csv():
#     csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
#     return max(csvs, key=os.path.getmtime) if csvs else None


# def empty_timeseries_fig():
#     fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.12)
#     for i, col in enumerate(TARGET_COLUMNS):
#         r, c = divmod(i, 2)
#         r, c = r + 1, c + 1
#         fig.add_trace(
#             go.Scatter(x=[], y=[], mode="lines", name=col, line=dict(width=1.5)),
#             row=r,
#             col=c,
#         )
#         fig.update_xaxes(title_text="Sample", row=r, col=c)
#         fig.update_yaxes(title_text=col, row=r, col=c)
#     return fig


# def empty_pred_fig():
#     fig = go.Figure()
#     for cls in PRED_CLASSES:
#         fig.add_trace(go.Bar(x=["Prediction"], y=[0], name=cls, showlegend=False))
#     fig.update_layout(
#         barmode="stack",
#         yaxis=dict(range=[0, 1], title="Vote Ratio", tickformat=".0%"),
#         xaxis=dict(showticklabels=False),
#         height=CHART_HEIGHT,
#         title=dict(text="Model Majority Vote", x=0.5),
#         showlegend=False,
#     )
#     return fig


# def build_pred_fig_from_votes(vote_counts: dict):
#     total = sum(vote_counts.values()) or 1
#     best = max(vote_counts, key=vote_counts.get)
#     fig = go.Figure()
#     for cls, count in vote_counts.items():
#         ratio = count / total
#         color = "green" if cls == best else "lightgray"
#         fig.add_trace(
#             go.Bar(
#                 x=["Prediction"],
#                 y=[ratio],
#                 name=cls,
#                 marker=dict(color=color),
#                 showlegend=False,
#                 hovertemplate=f"{cls}: {ratio:.1%}<extra></extra>",
#             )
#         )
#     fig.update_layout(
#         barmode="stack",
#         yaxis=dict(range=[0, 1], title="Vote Ratio", tickformat=".0%"),
#         xaxis=dict(showticklabels=False),
#         height=CHART_HEIGHT,
#         title=dict(text=f"Majority Vote — {best}", x=0.5),
#     )
#     return fig


# def preprocess(df: pd.DataFrame):
#     """
#     """
#     if not isinstance(df, pd.DataFrame):
#         df = pd.DataFrame(df)

#     df = df.select_dtypes(include=[np.number]).iloc[:, :4].copy()

=======
#     if not isinstance(df, pd.DataFrame):
#         df = pd.DataFrame(df)


#     df = df.select_dtypes(include=[np.number]).iloc[:, :4].copy()


>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#     if df.empty or len(df) < 80:
#         print(f"[WARN] CSV invalid: df shape={df.shape}")
#         return None

#     # ==== Step 1: diff_data_like ====
#     try:
#         if hasattr(df, "diff"):
<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#         else:
#             print("[WARN] df has no diff method; skipping diff")
#     except Exception as e:
#         print(f"[WARN] diff_data_like failed: {e}")
#         return None

#     # ==== Step 2: highpass_fft_batch ====
#     try:
#         df_np = df.to_numpy(dtype=np.float32)[None, :, :]
#         df_np = highpass_fft_batch(df_np)
#         df_np = df_np[0]
#     except Exception as e:
#         print(f"[WARN] highpass_fft_batch failed: {e}")
#         return None

#     # ==== Step 3: standard scaling ====
#     try:
#         df_np = SCALER.fit_transform(df_np)
#     except Exception as e:
#         print(f"[WARN] SCALER failed: {e}")
#         return None

#     return df_np


# # ===================== DASH =====================
# app = Dash(__name__)
# app.layout = html.Div(
#     style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui"},
#     children=[
#         html.H3("Live Gas Readings + Model Predictions", style={"marginBottom": "6px"}),
#         html.Div(id="file-indicator", style={"marginBottom": "12px"}),
#         html.Div(
#             style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px"},
#             children=[
#                 dcc.Graph(
#                     id="timeseries-graph",
#                     figure=empty_timeseries_fig(),
#                     config={"displayModeBar": False},
#                 ),
#                 dcc.Graph(
#                     id="pred-graph",
#                     figure=empty_pred_fig(),
#                     config={"displayModeBar": False},
#                 ),
#             ],
#         ),
#         dcc.Interval(id="tick-fast", interval=1_000, n_intervals=0),
#         dcc.Interval(id="tick-slow", interval=5_000, n_intervals=0),
#     ],
# )


# # ---- FAST: update signals ----
# @app.callback(
#     Output("timeseries-graph", "figure"),
#     Output("file-indicator", "children"),
#     Input("tick-fast", "n_intervals"),
# )
# def update_timeseries(_n):
#     latest = find_latest_csv()
#     if not latest:
#         return empty_timeseries_fig(), "Waiting for CSV..."
#     try:
#         df = pd.read_csv(latest)
#     except Exception as e:
#         return empty_timeseries_fig(), f"Read error: {e}"
#     tail = df.tail(10000)
#     fig = empty_timeseries_fig()
#     for i, col in enumerate(TARGET_COLUMNS):
#         if col in tail.columns:
#             y = pd.to_numeric(tail[col], errors="coerce")
#             mask = y.notna()
#             fig.data[i].x = list(range(mask.sum()))
#             fig.data[i].y = list(y[mask])
#     return fig, f"Using: {Path(latest).name}"


# # ---- SLOW: update prediction (every 5s) ----
# @app.callback(
#     Output("pred-graph", "figure"),
#     Input("tick-slow", "n_intervals"),
# )
# def update_predictions(_n):
#     latest = find_latest_csv()
#     if not latest:
#         return empty_pred_fig()

#     try:
#         df = pd.read_csv(latest)
#     except Exception:
#         return empty_pred_fig()

#     arr = preprocess(df)
#     if arr is None or len(arr) < WINDOW_SIZE:
#         return empty_pred_fig()

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#     windows = [
#         arr[i : i + WINDOW_SIZE] for i in range(0, len(arr) - WINDOW_SIZE + 1, STRIDE)
#     ]
#     preds = []
#     with torch.no_grad():
#         for w in windows:
#             X = torch.tensor(w, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#             logits = MODEL(X)
#             preds.append(int(torch.argmax(logits, dim=1).cpu()))

<<<<<<< HEAD
=======

>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
#     idx2cls = {v: k for k, v in CLASS_LABELS.items()}
#     counts = {cls: 0 for cls in idx2cls.values()}
#     for p in preds:
#         counts[idx2cls.get(p, "unknown")] += 1
#     # counts = {cls: 0 for cls in PRED_CLASSES}
#     # for p in preds:
#     #     if p < len(PRED_CLASSES):
#     #         counts[PRED_CLASSES[p]] += 1

#     return build_pred_fig_from_votes(counts)


# if __name__ == "__main__":
#     app.run(debug=True)

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