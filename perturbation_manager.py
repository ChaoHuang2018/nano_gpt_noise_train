import torch
import torch.nn as nn

# ==== 1. 扰动注入器 ====
class PerturbationInjector(nn.Module):
    def __init__(self, forward_eps=1e-3, grad_eps=1e-3, mode="none", name="unnamed"):
        super().__init__()
        self.forward_eps = forward_eps
        self.grad_eps = grad_eps
        self.mode = mode
        self.name = name

    def forward(self, x):
        if self.mode in ["forward", "both"]:
            noise = torch.empty_like(x).uniform_(-self.forward_eps, self.forward_eps)
            x = x + noise
        if self.mode in ["backward", "both"] and x.requires_grad:
            def _hook(grad):
                grad_noise = torch.empty_like(grad).uniform_(-self.grad_eps, self.grad_eps)
                return grad + grad_noise
            x.register_hook(_hook)
        return x
    

# ==== 3. 控制扰动模式 ====
def set_perturbation_mode(model, mode):
    for m in model.modules():
        if isinstance(m, PerturbationInjector):
            m.mode = mode
