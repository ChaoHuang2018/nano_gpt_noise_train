import os
import subprocess
import csv
from itertools import product
import pandas as pd
import numpy as np

def normalize(val, decimals=8):
    return round(float(val), decimals)

# ==== 配置搜索空间 ====
modes = ["both"]
epsilons = np.logspace(-6, -5, num=3).tolist()
layer_types = ["none"]
rates = np.logspace(-4, -2, num=5).tolist()
REPEATS = 10

# ==== 输出文件 ====
summary_dir = "summaries"
os.makedirs(summary_dir, exist_ok=True)
summary_csv_path = os.path.join(summary_dir, "summary_NPU_lr_scan.csv")
fail_log_path = os.path.join(summary_dir, "failed_runs_lr_scan.log")

# ==== 读取已完成项 ====
completed_runs = set()
if os.path.exists(summary_csv_path):
    print(f"Loading existing summary from {summary_csv_path}")
    df_existing = pd.read_csv(summary_csv_path)
    for _, row in df_existing.iterrows():
        key = (
            int(row["repeat_id"]),
            row["error_mode"],
            normalize(row["forward_eps"]),
            normalize(row["grad_eps"]),
            str(row["layer_types"]),
            normalize(row["learning_rate"])
        )
        completed_runs.add(key)

# ==== 准备写入 summary ====
file_exists = os.path.exists(summary_csv_path)
file_empty = (not file_exists) or os.path.getsize(summary_csv_path) == 0

headers = [
    "repeat_id",
    "error_mode", "forward_eps", "grad_eps", "layer_types", "learning_rate",
    "final_loss"
]
try:
    iter_cols = [f"loss_iter_{i}" for i in range(501)]
    headers += iter_cols
except Exception:
    iter_cols = []

if file_empty:
    with open(summary_csv_path, "w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(headers)

summary_file = open(summary_csv_path, "a", newline="")
summary_writer = csv.writer(summary_file)

# ==== 遍历 learning rate，在每个 learning rate 下先做 baseline ====
for lr in rates:

    baseline_name = f"clean_baseline_lr{lr}"
    baseline_out_dir = os.path.join("out-shakespeare-char", baseline_name)
    baseline_loss_path = os.path.join(baseline_out_dir, "final_train_loss.txt")
    baseline_loss_log_path = os.path.join(baseline_out_dir, "train_loss_log.csv")
    loss_values = []

    baseline_key = (0, "none", 0.0, 0.0, str([]), normalize(lr))

    if baseline_key in completed_runs:
        print(f"✅ Baseline already exists for learning rate {lr}")
    else:
        print(f"🚀 Running baseline training for learning rate {lr}...")
        subprocess.run([
            "python", "train.py",
            "config/train_shakespeare_char.py",
            "--init_from=scratch",
            f"--init_checkpoint_dir=out-shakespeare-char/clean_init",
            f"--out_dir=out-shakespeare-char/{baseline_name}",
            f"--error_mode=none",
            f"--forward_eps=0.0",
            f"--grad_eps=0.0",
            f"--error_layer_types=",
            f"--learning_rate={lr}",
            f"--run_name=baseline_lr{lr}"
        ], check=True)

        with open(baseline_loss_path) as f:
            _, loss_str = f.readline().strip().split(",")
            final_loss = float(loss_str)

        try:
            df_log = pd.read_csv(baseline_loss_log_path)
            loss_values = df_log["train_loss"].tolist()
        except Exception as e:
            print(f"  ⚠️  Failed to read train_loss_log.csv for baseline_lr{lr}: {e}")
            loss_values = []

        with open(summary_csv_path, "a", newline="") as fsum:
            writer = csv.writer(fsum)
            writer.writerow([
                0, "none", 0.0, 0.0, [], lr, final_loss,
                *loss_values
            ])
        print(f"✅ Baseline training complete for learning rate {lr}. Final loss: {final_loss:.4f}")

    # ==== 遍历扰动配置 ====
    for mode, eps, layers in product(modes, epsilons, layer_types):
        layer_str = layers
        print(f"\n[Config: mode={mode}, eps={eps}, layers={layer_str}, lr={lr}]")

        for repeat_id in range(REPEATS):
            run_key = (
                repeat_id,
                mode,
                normalize(eps),
                normalize(eps),
                str(layer_str),
                normalize(lr)
            )

            if run_key in completed_runs:
                print(f"  Skipping: repeat={repeat_id}, mode={mode}, eps={eps}, layer={layer_str}, lr={lr}")
                continue

            run_name = f"{mode}_{eps}_{layer_str}_lr{lr}_rep{repeat_id}".replace(".", "")
            out_dir = os.path.join("out-shakespeare-char", run_name)

            try:
                subprocess.run([
                    "python", "train.py",
                    "config/train_shakespeare_char.py",
                    "--init_from=scratch",
                    f"--init_checkpoint_dir=out-shakespeare-char/clean_init",
                    f"--out_dir=out-shakespeare-char/{run_name}",
                    f"--error_mode={mode}",
                    f"--forward_eps={eps}",
                    f"--grad_eps={eps}",
                    f"--error_layer_types={layers}",
                    f"--learning_rate={lr}",
                    f"--run_name={run_name}"
                ], check=True)

                loss_path = os.path.join(out_dir, "final_train_loss.txt")
                with open(loss_path) as f:
                    _, loss_str = f.readline().strip().split(",")
                    final_loss = float(loss_str)

                loss_log_path = os.path.join(out_dir, "train_loss_log.csv")
                loss_values = []

                try:
                    df_log = pd.read_csv(loss_log_path)
                    loss_values = df_log["train_loss"].tolist()
                except Exception as e:
                    print(f"  ⚠️  Failed to read train_loss_log.csv for {run_name}: {e}")
                    loss_values = []

                summary_writer.writerow([
                    repeat_id,
                    mode, eps, eps, layer_str, lr, final_loss,
                    *loss_values
                ])
                summary_file.flush()

            except Exception as e:
                print(f"  ❌ Run failed: repeat={repeat_id}, mode={mode}, eps={eps}, layer={layer_str}, lr={lr}")
                with open(fail_log_path, "a") as ferr:
                    ferr.write(f"{repeat_id},{mode},{eps},{layer_str},{lr}: {e}\n")

summary_file.close()
print("\n✅ All experiments completed or skipped as needed.")
