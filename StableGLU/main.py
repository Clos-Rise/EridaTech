import torch
import torch.nn as nn


class StableGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, gate_eps=1e-4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * dim * 2 / 3)
        self.eps = gate_eps
        self.gate_proj = nn.Linear(dim, 1, bias=False)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.normal_(self.value_proj.weight, std=0.01)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def forward(self, x):
        s = self.gate_proj(x)
        gate = 0.5 + 0.5 * s * torch.rsqrt(1.0 + s * s + self.eps)
        v = self.value_proj(x)
        v = v * torch.rsqrt(1.0 + 0.01 * v * v)
        return self.dropout(self.out_proj(v * gate))
