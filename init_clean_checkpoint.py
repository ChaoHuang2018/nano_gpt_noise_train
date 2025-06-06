import os
import torch
from model import GPT, GPTConfig

# 以下是NPU需要
import torch_npu 
from torch_npu.contrib import transfer_to_npu

# ==== 配置 ====
init_from = "scratch"  # or "gpt2"
out_dir = "out-shakespeare-char/clean_init"
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "ckpt_clean.pt")

# ==== 初始化模型 ====
if init_from == "gpt2":
    print("[✓] Loading GPT-2 pretrained weights...")
    model = GPT.from_pretrained("gpt2")
    model_args = {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "block_size": model.config.block_size,
        "bias": model.config.bias,
        "vocab_size": model.config.vocab_size,
    }
elif init_from == "scratch":
    print("[✓] Initializing GPT model from scratch")
    config = GPTConfig(
        block_size=256,
        vocab_size=65,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        bias=False
    )
    model = GPT(config)
    model_args = config.__dict__
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# ==== 保存 ckpt ====
ckpt = {
    "model": model.state_dict(),
    "model_args": model_args,
    "iter_num": 0,
    "best_val_loss": 1e9,
    "config": {},  # 可选：额外配置
}
torch.save(ckpt, save_path)
print(f"[✓] Clean initialization checkpoint saved to {save_path}")