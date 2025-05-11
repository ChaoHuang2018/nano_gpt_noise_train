import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out-shakespeare-char/train_loss_log_20250511_042845.csv")

plt.plot(df["iter"], df["train_loss_clean"], label="Clean Train Loss")
plt.plot(df["iter"], df["train_loss_noisy"], label="Noisy Train Loss")
plt.plot(df["iter"], df["delta_train_loss"], label="Δ Loss (Noisy - Clean)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Clean vs Noisy Loss Comparison")
plt.savefig("out-shakespeare-char/loss_plot.png")

plt.show()

