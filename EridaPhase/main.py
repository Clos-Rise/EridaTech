import torch
import torch.nn as nn
import torch.nn.functional as F


class EridiumPhase(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult=4, window_size=16, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.eps = eps
        self._cached_mask = None
        self._cached_len = None

        self.phase_router = nn.Linear(dim, 1, bias=True)
        nn.init.normal_(self.phase_router.weight, std=0.1)
        nn.init.zeros_(self.phase_router.bias)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.bypass = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )
        nn.init.normal_(self.bypass[1].weight, std=0.001)

        hidden = int(dim * ffn_mult * 2 / 3)
        self.ffn_up = nn.Linear(dim, hidden, bias=False)
        self.ffn_down = nn.Linear(hidden, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.normal_(self.ffn_up.weight, std=0.02)
        nn.init.normal_(self.ffn_down.weight, std=0.02)

    def _phase_gate(self, phase):
        return 0.5 + 0.5 * (phase / torch.sqrt(1 + phase ** 2))

    def _get_local_mask(self, N, device):
        if self._cached_mask is None or self._cached_len != N or self._cached_mask.device != device:
            mask = torch.zeros(N, N, device=device)
            for i in range(N):
                start = max(0, i - self.window_size // 2)
                end = min(N, i + self.window_size // 2 + 1)
                mask[i, start:end] = 1.0
            self._cached_mask = mask
            self._cached_len = N
        return self._cached_mask

    def forward(self, x, return_phase=False):
        B, N, D = x.shape
        phase_raw = self.phase_router(x).squeeze(-1)
        phase = self._phase_gate(phase_raw)

        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_global = torch.softmax(scores, dim=-1)
        out_global = torch.matmul(attn_global, v)

        local_mask = self._get_local_mask(N, x.device)
        scores_local = scores.masked_fill(local_mask[None, None, ...] == 0, float('-inf'))
        attn_local = torch.softmax(scores_local, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        out_local = torch.matmul(attn_local, v)

        phase_h = phase.view(B, 1, N, 1)
        attn_out = out_global * phase_h + out_local * (1 - phase_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out(attn_out)

        bypass_out = self.bypass(x)
        bypass_out = bypass_out * torch.rsqrt(1 + 0.01 * bypass_out ** 2)

        gate_attn = phase.view(B, N, 1)
        x = residual + gate_attn * attn_out + (1 - gate_attn) * bypass_out
        x = self.norm_out(x)

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn_down(F.gelu(self.ffn_up(x_norm)))
        ffn_out = ffn_out * torch.rsqrt(1 + 0.01 * ffn_out ** 2)
        gate_ffn = phase.view(B, N, 1)
        x = residual + gate_ffn * ffn_out + (1 - gate_ffn) * (x_norm * 0.1)

        if return_phase:
            return x, phase
        return x
