import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 参数设置 =====
folder = "demo_data"  # 改成你的csv文件夹路径
save_path = "demo_data/all_csv_plots.png"  # 输出图片

# ===== 读取所有CSV文件 =====
csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

n_files = len(csv_files)
fig, axes = plt.subplots(n_files, 1, figsize=(10, 4 * n_files), squeeze=False)
axes = axes.flatten()

for i, filename in enumerate(csv_files):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)

    # 删除全空列
    df = df.dropna(axis=1, how="all")

    # 删除第一列（假设是timestep）
    if df.shape[1] > 1:
        df = df.iloc[:, 1:]
    if "Benzene" in df.columns:
        df = df.drop(columns=["Benzene"])
    # 绘制
    df.plot(ax=axes[i])
    axes[i].set_title(filename)
    axes[i].set_xlabel("Seconds")
    axes[i].set_ylabel("Value")

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
