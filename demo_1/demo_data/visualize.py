import os
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
folder = "demo_data"
save_path = "demo_data/all_csv_plots.png"

=======

folder = "demo_data"  
save_path = "demo_data/all_csv_plots.png"  


>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

n_files = len(csv_files)
fig, axes = plt.subplots(n_files, 1, figsize=(10, 4 * n_files), squeeze=False)
axes = axes.flatten()

for i, filename in enumerate(csv_files):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)

<<<<<<< HEAD
    df = df.dropna(axis=1, how="all")

=======
    
    df = df.dropna(axis=1, how="all")

    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    if df.shape[1] > 1:
        df = df.iloc[:, 1:]
    if "Benzene" in df.columns:
        df = df.drop(columns=["Benzene"])
<<<<<<< HEAD
=======
    
>>>>>>> df11f5f239fc2fdc366756d13eab7f602ee3f235
    df.plot(ax=axes[i])
    axes[i].set_title(filename)
    axes[i].set_xlabel("Seconds")
    axes[i].set_ylabel("Value")

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()