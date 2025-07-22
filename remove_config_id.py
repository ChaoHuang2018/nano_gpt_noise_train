import pandas as pd
import shutil

summary_path = "summaries/summary_NPU.csv"
backup_path = "summaries/summary_NPU_backup.csv"

# 备份
shutil.copyfile(summary_path, backup_path)
print(f"Backup created at {backup_path}")

# 读取并删除 config_id 列
df = pd.read_csv(summary_path)
df = df.drop(columns=["config_id"])
df.to_csv(summary_path, index=False)
print(f"config_id column removed from {summary_path}")