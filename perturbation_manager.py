import torch
import torch.nn as nn

# ==== 1. 扰动注入器 ====
class PerturbationInjector(nn.Module):
    def __init__(self, forward_eps=1e-3, grad_eps=1e-3, mode="none"):
        super().__init__()
        self.forward_eps = forward_eps
        self.grad_eps = grad_eps
        self.mode = mode

    def forward(self, x):
        if self.mode in ["forward", "both"] and not self.training:
            noise = torch.empty_like(x).uniform_(-self.forward_eps, self.forward_eps)
            x = x + noise
        if self.mode in ["backward", "both"] and x.requires_grad:
            def _hook(grad):
                grad_noise = torch.empty_like(grad).uniform_(-self.grad_eps, self.grad_eps)
                return grad + grad_noise
            x.register_hook(_hook)
        return x

# ==== 2. 注入包装器：在 Dropout 前注入扰动 ====
def insert_perturbation_before_dropout(model, mode="none", forward_eps=1e-3, grad_eps=1e-3):
    def wrap_mlp(module):
        perturb = PerturbationInjector(mode, forward_eps, grad_eps)
        def new_forward(x):
            x = module.c_fc(x)
            x = module.gelu(x)
            x = module.c_proj(x)
            x = perturb(x)
            x = module.dropout(x)
            return x
        module.forward = new_forward

    def wrap_attention(module):
        perturb = PerturbationInjector(mode, forward_eps, grad_eps)
        def new_forward(x):
            B, T, C = x.size()
            q, k, v = module.c_attn(x).split(module.n_embd, dim=2)
            q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            v = v.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
            
            if module.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=module.dropout if module.training else 0, is_causal=True)
            else:

                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
                att = att.masked_fill(module.bias[:, :, :T, :T] == 0, float('-inf'))
                att = nn.functional.softmax(att, dim=-1)
                att = module.attn_dropout(att)
                y = att @ v
                
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = module.c_proj(y)
                
            y = perturb(y)
            y = module.resid_dropout(y)
            return y
        module.forward = new_forward

    def wrap_layernorm(module):
        perturb = PerturbationInjector(mode, forward_eps, grad_eps)
        def new_forward(x):
            x = nn.functional.layer_norm(
                x, module.weight.shape, module.weight, module.bias, 1e-5
            )
            x = perturb(x)
            return x
        module.forward = new_forward

    for name, submodule in model.named_modules():
        if submodule.__class__.__name__ == "MLP":
            wrap_mlp(submodule)
        elif submodule.__class__.__name__ == "CausalSelfAttention":
            wrap_attention(submodule)
        elif submodule.__class__.__name__ == "LayerNorm":
            wrap_layernorm(submodule)

# ==== 3. 控制扰动模式 ====
def set_perturbation_mode(model, mode):
    for m in model.modules():
        if isinstance(m, PerturbationInjector):
            m.mode = mode