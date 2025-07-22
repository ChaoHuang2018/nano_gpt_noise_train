import os
import subprocess
import csv
from itertools import product
import pandas as pd
import numpy as np

def normalize_float(val, decimals=8):
    return round(float(val), decimals)

def ensure_init_checkpoint(n_layer: int):
    init_dir = f"out-shakespeare-char/clean_init_layer{n_layer}"
    ckpt_path = os.path.join(init_dir, "ckpt_clean.pt")
    if os.path.exists(ckpt_path):
        print(f"✅ Init checkpoint already exists: {ckpt_path}")
        return
    print(f"🚀 Generating init checkpoint for n_layer={n_layer}")
    os.makedirs(init_dir, exist_ok=True)
    
    result = subprocess.run([
        "python", "init_clean_checkpoint.py",
        "--n_layer", str(n_layer),
        "--out_dir", init_dir
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate init checkpoint for n_layer={n_layer}")
    

# ==== 配置 ====
errors = np.logspace(-7, -5, num=5).tolist()
learning_rates = np.logspace(-4, -3, num=3).tolist()
layer_counts = [4, 6, 8, 10, 12]
REPEATS = 5

summary_dir = "summaries"
os.makedirs(summary_dir, exist_ok=True)
summary_csv_path = os.path.join(summary_dir, "summary_NPU_block_test.csv")
fail_log_path = os.path.join(summary_dir, "failed_block_test.log")

completed_keys = set()
if os.path.exists(summary_csv_path):
    df_existing = pd.read_csv(summary_csv_path)
    for _, row in df_existing.iterrows():
        key = (
            int(row["repeat_id"]),
            normalize_float(row["error_eps"]),
            normalize_float(row["learning_rate"]),
            int(row["n_layer"])
        )
        completed_keys.add(key)

headers = ["repeat_id", "error_eps", "learning_rate", "n_layer", "final_loss"]
iter_cols = [f"loss_iter_{i}" for i in range(501)]
headers += iter_cols

if not os.path.exists(summary_csv_path) or os.path.getsize(summary_csv_path) == 0:
    with open(summary_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)

summary_file = open(summary_csv_path, "a", newline="")
summary_writer = csv.writer(summary_file)

# ==== 主循环 ====
for lr, n_layer in product(learning_rates, layer_counts):

    ensure_init_checkpoint(n_layer)
    init_ckpt_dir = f"out-shakespeare-char/clean_init_layer{n_layer}"

    # === Baseline ===
    baseline_key = (0, 0.0, normalize_float(lr), n_layer)
    if baseline_key not in completed_keys:
        run_name = f"blocktest_baseline_lr{lr}_L{n_layer}".replace(".", "")
        out_dir = os.path.join("out-shakespeare-char", run_name)

        try:
            print(f"🚀 Running baseline: lr={lr}, n_layer={n_layer}")
            subprocess.run([
                "python", "train.py",
                "config/train_shakespeare_char.py",
                "--init_from=scratch",
                f"--init_checkpoint_dir={init_ckpt_dir}",
                f"--out_dir={out_dir}",
                "--error_mode=none",
                "--forward_eps=0.0",
                "--grad_eps=0.0",
                f"--error_layer_types=",
                f"--learning_rate={lr}",
                f"--n_layer={n_layer}",
                f"--run_name={run_name}"
            ], check=True)

            with open(os.path.join(out_dir, "final_train_loss.txt")) as f:
                _, loss_str = f.readline().strip().split(",")
                final_loss = float(loss_str)

            loss_values = []
            try:
                df_log = pd.read_csv(os.path.join(out_dir, "train_loss_log.csv"))
                loss_values = df_log["train_loss"].tolist()
            except Exception as e:
                print(f"⚠️ Failed to read train_loss_log.csv for baseline: {e}")

            summary_writer.writerow([
                0, 0.0, lr, n_layer, final_loss, *loss_values
            ])
            summary_file.flush()

        except Exception as e:
            print(f"❌ Baseline run failed: lr={lr}, n_layer={n_layer}")
            with open(fail_log_path, "a") as ferr:
                ferr.write(f"baseline,0,0.0,{lr},{n_layer}: {e}\n")
    else:
        print(f"✅ Baseline already done: lr={lr}, n_layer={n_layer}")

    # === 实验部分 ===
    for eps in errors:
        for repeat_id in range(REPEATS):
            key = (repeat_id, normalize_float(eps), normalize_float(lr), n_layer)
            if key in completed_keys:
                print(f"✅ Skip repeat {repeat_id}, eps={eps}, lr={lr}, L={n_layer}")
                continue

            run_name = f"blocktest_eps{eps}_lr{lr}_L{n_layer}_rep{repeat_id}".replace(".", "")
            out_dir = os.path.join("out-shakespeare-char", run_name)

            try:
                print(f"🚀 Running: eps={eps}, lr={lr}, n_layer={n_layer}, rep={repeat_id}")
                subprocess.run([
                    "python", "train.py",
                    "config/train_shakespeare_char.py",
                    "--init_from=scratch",
                    f"--init_checkpoint_dir={init_ckpt_dir}",
                    f"--out_dir={out_dir}",
                    "--error_mode=both",
                    f"--forward_eps={eps}",
                    f"--grad_eps={eps}",
                    "--error_layer_types=none",
                    f"--learning_rate={lr}",
                    f"--n_layer={n_layer}",
                    f"--run_name={run_name}"
                ], check=True)

                with open(os.path.join(out_dir, "final_train_loss.txt")) as f:
                    _, loss_str = f.readline().strip().split(",")
                    final_loss = float(loss_str)

                loss_values = []
                try:
                    df_log = pd.read_csv(os.path.join(out_dir, "train_loss_log.csv"))
                    loss_values = df_log["train_loss"].tolist()
                except Exception as e:
                    print(f"⚠️ Failed to read train_loss_log.csv for eps={eps}, lr={lr}: {e}")

                summary_writer.writerow([
                    repeat_id, eps, lr, n_layer, final_loss, *loss_values
                ])
                summary_file.flush()

            except Exception as e:
                print(f"❌ Run failed: eps={eps}, lr={lr}, L={n_layer}, rep={repeat_id}")
                with open(fail_log_path, "a") as ferr:
                    ferr.write(f"{repeat_id},{eps},{lr},{n_layer}: {e}\n")

summary_file.close()
print("✅ All block experiments completed.")
