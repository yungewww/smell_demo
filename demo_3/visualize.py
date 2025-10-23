import os
import pandas as pd
import matplotlib.pyplot as plt

# ========== 配置 ==========
DATA_DIR = "data/train"
SAVE_DIR = "visuals"
os.makedirs(SAVE_DIR, exist_ok=True)

Y_MIN, Y_MAX = -100, 1000

# ========== 主逻辑 ==========
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue

    csv_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(csv_path)

    # 创建大图 (3行 x 4列 = 12个小图)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(df.columns[:12]):
        axes[i].plot(df[col], linewidth=0.8)
        axes[i].set_title(col, fontsize=8)
        axes[i].set_ylim(Y_MIN, Y_MAX)
        axes[i].set_xticks([])

    # 去掉多余子图（防止有空）
    for j in range(len(df.columns), 12):
        axes[j].axis("off")

    fig.suptitle(filename, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(SAVE_DIR, filename.replace(".csv", ".jpg"))
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {save_path}")
