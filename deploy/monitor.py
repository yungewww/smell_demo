import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import argparse
from pathlib import Path
import json
from run import format_stimuli

# === CONFIGURATION ===
config_path = "config.json"
with open(config_path, "r") as c:
    config = json.load(c)

DATA_DIR = os.path.join(config["data_dir"], format_stimuli(config))
TARGET_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
WINDOW_SIZE = 10000  # samples shown (tail)


def find_latest_csv():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)


# === Setup 2x3 subplots (one per channel) ===
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=110, sharex=False)
axes = axes.flatten()

# Map each target column to its axis and an empty Line2D
ax_map = {}
line_map = {}
for ax, col in zip(axes, TARGET_COLUMNS):
    ax.set_title(col)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    # initialize empty line
    (ln,) = ax.plot([], [], linewidth=1.3, alpha=0.9)
    ax_map[col] = ax
    line_map[col] = ln

# If fewer axes than targets (shouldn't happen here), hide extras
for j in range(len(TARGET_COLUMNS), len(axes)):
    axes[j].axis("off")

last_file_shown = [None]  # mutable for closure


def animate(_frame_idx):
    latest_csv = find_latest_csv()
    if not latest_csv or not os.path.exists(latest_csv):
        # Show "waiting" message on the first axis only; clear lines
        for col in TARGET_COLUMNS:
            line_map[col].set_data([], [])
            ax_map[col].relim()
            ax_map[col].autoscale_view()
        axes[0].set_title("Waiting for CSV file...")
        return

    # Update figure suptitle when file changes
    if latest_csv != last_file_shown[0]:
        fig.suptitle(f"Live Gas Readings — {os.path.basename(latest_csv)}", fontsize=12)
        last_file_shown[0] = latest_csv

    try:
        df = pd.read_csv(latest_csv)
    except Exception as e:
        # If file is mid-write, just skip this tick
        print("Read error:", e)
        return

    if df.empty:
        for col in TARGET_COLUMNS:
            line_map[col].set_data([], [])
            ax_map[col].relim()
            ax_map[col].autoscale_view()
        return

    # Tail window
    tail = df.tail(WINDOW_SIZE).copy()
    x = pd.RangeIndex(start=0, stop=len(tail))

    # For each channel, coerce numeric and update its line & axes limits
    for col in TARGET_COLUMNS:
        ax = ax_map[col]
        ln = line_map[col]

        if col not in tail.columns:
            # If column missing, clear the line and annotate once
            ln.set_data([], [])
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"{col} (missing)")
            continue

        y = pd.to_numeric(tail[col], errors="coerce")
        mask = y.notna()
        if not mask.any():
            ln.set_data([], [])
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"{col} (no numeric data)")
            continue

        # Update line data
        ln.set_data(x[mask], y[mask])

        # Update axes limits to fit new data
        ax.relim()
        ax.autoscale_view()

        # Keep consistent x range [0, len-1] for readability
        ax.set_xlim(0, max(1, len(tail) - 1))

        # Keep the original clean title
        ax.set_title(col)


ani = animation.FuncAnimation(fig, animate, interval=1000)  # ms
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()