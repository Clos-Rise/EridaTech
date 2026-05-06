import torch
import torch.nn as nn
import math


class EridaGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * dim * 2 / 3)
        self.hidden_dim = hidden_dim
        self.tau = 9.0

        self.gate_proj = nn.Linear(dim, 1, bias=True)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.value_proj.weight, std=math.sqrt(2 / dim) * 0.3)
        nn.init.normal_(self.out_proj.weight, std=math.sqrt(2 / hidden_dim))

    def forward(self, x):
        s = self.gate_proj(x)
        gate = 0.5 + 0.5 * (s / torch.sqrt(self.tau + s * s))
        v = self.value_proj(x)
        h = v * gate
        return self.dropout(self.out_proj(h))
