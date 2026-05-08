import torch
import torch.nn as nn

class FastGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * dim * 2 / 3)
        self.gate_proj = nn.Linear(dim, 1, bias=False)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x):
        s = self.gate_proj(x)
        gate = 0.5 + 0.5 * (s * torch.rsqrt(1.0 + s * s))
        v = self.value_proj(x)
        return self.dropout(self.out_proj(v * gate))
