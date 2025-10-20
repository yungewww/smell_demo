# Serial Data Collection & Monitoring

## Quick Start

Modify config.json with the stimuli you plan to measure.

Open **two separate terminals** and run the following commands:

```bash
# Terminal 1 — Collect data to CSV
python3 run.py

# Terminal 2 — Launch the monitoring UI
python3 monitor.py

```

# Visualize a specific recorded file
python3 smell_sensors/csv_collect/iclr_visualize_offline.py -d ./data/bodai -g "bodai.5bb78b7cc030.csv"

# Or any file with "ambient" in the name
python3 smell_sensors/csv_collect/iclr_visualize_offline.py -d ./data/bodai -g "*bodai*.csv" --interactive

# Optional: smooth with a rolling mean and save a PNG
python3 smell_sensors/csv_collect/iclr_visualize_offline.py -d ./data -g "*ambient*.csv" -r 10 -s ./data/plots/ambient.png

# Optional: one subplot per CSV (nice for comparisons)
python3 smell_sensors/csv_collect/iclr_visualize_offline.py -d ./data -g "*ambient*.csv" --per-file 


data1001 offline visualization
# Overlay them on one figure
python3 smell_sensors/csv_collect/iclr_visualize_offline.py \
  -d ./data \
  -g "*garlic50_almond50*.csv" \
  --interactive

# One subplot per file (nice for side-by-side)
python3 smell_sensors/csv_collect/iclr_visualize_offline.py \
  -d ./data \
  -g "*garlic50_almond50*.csv" \
  --per-file \
  --interactive