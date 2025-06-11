import os
import subprocess
import csv
from itertools import product
import pandas as pd
import numpy as np

# ==== 配置搜索空间 ====
# modes = ["forward", "both"]
modes = ["forward", "both"]
epsilons = np.logspace(-5, -1, num=9).tolist()
# none表示不指定扰动层，所有层都有扰动误差
layer_types = ["none", "MLP", "LayerNorm", "CausalSelfAttention"]
# repeats_per_config = 1
REPEATS = 20

# ==== 输出文件 ====
summary_dir = "summaries"
os.makedirs(summary_dir, exist_ok=True)
summary_csv_path = os.path.join(summary_dir, "summary_GPU.csv")
fail_log_path = os.path.join(summary_dir, "failed_runs.log")

# ==== 读取已完成项 ====
completed_runs = set()
if os.path.exists(summary_csv_path):
    print(f"Loading existing summary from {summary_csv_path}")
    df_existing = pd.read_csv(summary_csv_path)
    for _, row in df_existing.iterrows():
        completed_runs.add((int(row["config_id"]), int(row["repeat_id"])))
        
# ==== 准备写入 summary ====
file_exists = os.path.exists(summary_csv_path)
file_empty = (not file_exists) or os.path.getsize(summary_csv_path) == 0

headers = [
    "config_id", "repeat_id",
    "error_mode", "forward_eps", "grad_eps", "layer_types",
    "final_loss"
]
try:
    iter_cols = [f"loss_iter_{i}" for i in range(501)]
    headers += iter_cols
except Exception:
    iter_cols = []

# 只在文件不存在或为空时写表头
if file_empty:
    with open(summary_csv_path, "w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(headers)

# 后续追加写入
summary_file = open(summary_csv_path, "a", newline="")
summary_writer = csv.writer(summary_file)

# ==== 检查并运行 baseline（无扰动）（repeat_id = -1）====
baseline_name = "clean_baseline"
baseline_out_dir = "out-shakespeare-char/clean_baseline"

baseline_loss_path = os.path.join(baseline_out_dir, "final_train_loss.txt")

baseline_loss_log_path = os.path.join(baseline_out_dir, "train_loss_log.csv")
loss_values = []

baseline_recorded = False
if os.path.exists(summary_csv_path):
    df_existing = pd.read_csv(summary_csv_path)
    if -1 in df_existing["config_id"].values:
        print("✅ Baseline already exists in summary.")
        baseline_recorded = True

if not baseline_recorded:
    print("🚀 Running baseline training (no perturbation)...")
    subprocess.run([
            "python", "train.py",
            "config/train_shakespeare_char.py",
            "--init_from=scratch",
            f"--init_checkpoint_dir=out-shakespeare-char/clean_init",  # 用于加载 ckpt
            f"--out_dir=out-shakespeare-char/{baseline_name}",              # 用于保存结果
            f"--error_mode=none",
            f"--forward_eps=0.0",
            f"--grad_eps=0.0",
            f"--error_layer_types=",
            f"--run_name=baseline"
        ], check=True)

    with open(baseline_loss_path) as f:
        _, loss_str = f.readline().strip().split(",")
        final_loss = float(loss_str)
        
    try:
        df_log = pd.read_csv(baseline_loss_log_path)
        loss_values = df_log["train_loss"].tolist()
    except Exception as e:
        print(f"  ⚠️  Failed to read train_loss_log.csv for config BASELINE: {e}")
        loss_values = []

    with open(summary_csv_path, "a", newline="") as fsum:
        writer = csv.writer(fsum)
        writer.writerow([
            -1, 0,
            "none", 0.0, 0.0, [], final_loss,
            *loss_values
        ])
    print(f"✅ Baseline training complete. Final loss: {final_loss:.4f}")

# ==== 遍历配置并训练 ====
config_id = 0
for mode, eps, layers in product(modes, epsilons, layer_types):
    
    layer_str = layers  # e.g., "MLP"
    print(f"\n[Config {config_id}] mode={mode}, eps={eps}, layers={layer_str}")

    for repeat_id in range(REPEATS):
        if (config_id, repeat_id) in completed_runs:
            print(f"  Skipping repeat {repeat_id} (already completed)")
            continue

        run_name = f"{mode}_{eps}_{layer_str}_rep{repeat_id}".replace(".", "")
        out_dir = os.path.join("out-shakespeare-char", run_name)

        try:
            # === 调用训练脚本 ===
            subprocess.run([
                    "python", "train.py",
                    "config/train_shakespeare_char.py",
                    "--init_from=scratch",
                    f"--init_checkpoint_dir=out-shakespeare-char/clean_init",  # 用于加载 ckpt
                    f"--out_dir=out-shakespeare-char/{run_name}",              # 用于保存结果
                    f"--error_mode={mode}",
                    f"--forward_eps={eps}",
                    f"--grad_eps={eps}",
                    f"--error_layer_types={layers}",
                    f"--run_name={run_name}"
                ], check=True)

            # === 读取最终 loss ===
            loss_path = os.path.join(out_dir, "final_train_loss.txt")
            with open(loss_path) as f:
                _, loss_str = f.readline().strip().split(",")
                final_loss = float(loss_str)
            
            # === 读取最终 loss 完整记录 ===
            loss_log_path = os.path.join(out_dir, "train_loss_log.csv")
            loss_values = []

            try:
                df_log = pd.read_csv(loss_log_path)
                loss_values = df_log["train_loss"].tolist()
            except Exception as e:
                print(f"  ⚠️  Failed to read train_loss_log.csv for config {config_id} rep {repeat_id}: {e}")
                loss_values = []    
            

            # === 写入 summary ===
            summary_writer.writerow([
                config_id, repeat_id,
                mode, eps, eps, layer_str, final_loss,
                *loss_values
            ])
            summary_file.flush()

        except Exception as e:
            print(f"  ❌ Run failed: Config {config_id}, Repeat {repeat_id}")
            with open(fail_log_path, "a") as ferr:
                ferr.write(f"{config_id},{repeat_id},{mode},{eps},{layer_str}: {e}\n")

    config_id += 1

summary_file.close()
print("\n✅ All experiments completed or skipped as needed.")
    
# # ==== 写入 summary 文件头 ====
# with open(summary_csv_path, "w", newline="") as fsum:
#     writer = csv.writer(fsum)
#     writer.writerow([
#         "config_id", "repeat_id", 
#         "error_mode", "forward_eps", "grad_eps", "layer_types", 
#         "final_loss"
#     ])

#     config_id = 0

#     # ==== 枚举所有配置组合 ====
#     for mode, eps, layers in product(modes, epsilons, layer_types):
#         layer_str = layers
#         print(f"\n[Config {config_id}] mode={mode}, eps={eps}, layers={layer_str}")

#         for rep in range(repeats_per_config):
#             run_name = f"{mode}_{eps}_{layer_str}_rep{rep}".replace(".", "")
#             out_dir = os.path.join("out-shakespeare-char", run_name)

#             # === 调用训练脚本 ===
#             try:
#                 subprocess.run([
#                     "python", "train.py",
#                     "config/train_shakespeare_char.py",
#                     "--init_from=resume",
#                     f"--init_checkpoint_dir=out-shakespeare-char/clean_init",  # 用于加载 ckpt
#                     f"--out_dir=out-shakespeare-char/{run_name}",              # 用于保存结果
#                     f"--error_mode={mode}",
#                     f"--forward_eps={eps}",
#                     f"--grad_eps={eps}",
#                     f"--error_layer_types={layer_str}",
#                     f"--run_name={run_name}"
#                 ], check=True)

#                 # === 读取最终 loss ===
#                 loss_path = os.path.join(out_dir, "final_train_loss.txt")
#                 with open(loss_path) as f:
#                     _, loss_str = f.readline().strip().split(",")
#                     final_loss = float(loss_str)

#                 writer.writerow([
#                     config_id, rep,
#                     mode, eps, eps, layer_str,
#                     final_loss
#                 ])
#                 fsum.flush()

#             except Exception as e:
#                 print(f"[Run Failed] Config {config_id}, rep {rep}: {e}")
        
#         config_id += 1