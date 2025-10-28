import os
import pandas as pd
import matplotlib.pyplot as plt


folder = "demo_data"  
save_path = "demo_data/all_csv_plots.png"  


csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

n_files = len(csv_files)
fig, axes = plt.subplots(n_files, 1, figsize=(10, 4 * n_files), squeeze=False)
axes = axes.flatten()

for i, filename in enumerate(csv_files):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)

    
    df = df.dropna(axis=1, how="all")

    
    if df.shape[1] > 1:
        df = df.iloc[:, 1:]
    if "Benzene" in df.columns:
        df = df.drop(columns=["Benzene"])
    
    df.plot(ax=axes[i])
    axes[i].set_title(filename)
    axes[i].set_xlabel("Seconds")
    axes[i].set_ylabel("Value")

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()