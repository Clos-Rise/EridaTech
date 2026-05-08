import torch
import torch.nn as nn


class EridiumKV(nn.Module):
    def __init__(self, dim, num_heads, num_groups=2):
        super().__init__()
        assert num_heads % num_groups == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups

        self.k_shared = nn.Linear(dim, self.head_dim * num_groups, bias=False)
        self.v_shared = nn.Linear(dim, self.head_dim * num_groups, bias=False)

        self.k_transform = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.v_transform = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        for h in range(num_heads):
            nn.init.orthogonal_(self.k_transform[h])
            nn.init.orthogonal_(self.v_transform[h])

        self.k_shift = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.v_shift = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)

        self.k_blend = nn.Parameter(torch.zeros(num_heads))
        self.v_blend = nn.Parameter(torch.zeros(num_heads))

        nn.init.normal_(self.k_shared.weight, std=0.02)
        nn.init.normal_(self.v_shared.weight, std=0.02)

    def compress(self, x):
        B, S, D = x.shape
        k_base = self.k_shared(x).view(B, S, self.num_groups, self.head_dim)
        v_base = self.v_shared(x).view(B, S, self.num_groups, self.head_dim)
        return k_base, v_base

    def decompress(self, k_base, v_base):
        B, S, G, D = k_base.shape
        group_idx = torch.arange(self.num_heads, device=k_base.device) // self.heads_per_group
        k_base_h = k_base[:, :, group_idx, :]
        v_base_h = v_base[:, :, group_idx, :]

        k_spec = torch.einsum('bshd,hde->bshe', k_base_h, self.k_transform) + self.k_shift.view(1, 1, self.num_heads, self.head_dim)
        v_spec = torch.einsum('bshd,hde->bshe', v_base_h, self.v_transform) + self.v_shift.view(1, 1, self.num_heads, self.head_dim)

        kg = 0.5 + 0.5 * (self.k_blend / torch.sqrt(1 + self.k_blend ** 2))
        vg = 0.5 + 0.5 * (self.v_blend / torch.sqrt(1 + self.v_blend ** 2))

        k = k_base_h * (1 - kg.view(1, 1, -1, 1)) + k_spec * kg.view(1, 1, -1, 1)
        v = v_base_h * (1 - vg.view(1, 1, -1, 1)) + v_spec * vg.view(1, 1, -1, 1)

        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def forward(self, x):
        k_base, v_base = self.compress(x)
        return self.decompress(k_base, v_base)
