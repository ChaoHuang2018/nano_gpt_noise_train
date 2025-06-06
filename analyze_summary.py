import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==== 读取 summary ====
summary_path = "summaries/summary.csv"
df = pd.read_csv(summary_path)

# ==== 聚合统计 ====
group_cols = ["error_mode", "forward_eps", "layer_types"]
stats = df.groupby(group_cols)["final_loss"].agg(["mean", "std", "min", "max", "count"]).reset_index()

# ==== 保存统计结果 ====
stats_out_path = "summaries/summary_stats.csv"
stats.to_csv(stats_out_path, index=False)
print(f"[✓] 聚合统计保存至 {stats_out_path}")

# ==== 可视化 (可选) ====
# 为每组配置画一个 boxplot
plot_out_dir = "summaries/plots"
os.makedirs(plot_out_dir, exist_ok=True)

# 合成唯一配置 ID（用于分类）
df["config"] = df["error_mode"] + "_eps" + df["forward_eps"].astype(str) + "_" + df["layer_types"]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x="config", y="final_loss")
plt.xticks(rotation=45, ha="right")
plt.title("Loss Distribution per Perturbation Configuration")
plt.tight_layout()
plot_path = os.path.join(plot_out_dir, "boxplot_final_loss.png")
plt.savefig(plot_path)
print(f"[✓] Boxplot 图保存至 {plot_path}")
