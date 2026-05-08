import torch
import torch.nn as nn


class EPE(nn.Module):
    def __init__(self, num_heads, tau_init=8.0):
        super().__init__()
        self.slopes = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.tau = nn.Parameter(torch.ones(num_heads) * tau_init)

    def forward(self, seq_len_q, seq_len_k=None):
        if seq_len_k is None:
            seq_len_k = seq_len_q
        dist = torch.arange(seq_len_q)[:, None] - torch.arange(seq_len_k)[None, :]
        dist = dist.abs().float()
        slopes = self.slopes.abs().view(-1, 1, 1)
        tau = self.tau.abs().view(-1, 1, 1)
        return -slopes * dist / (1.0 + dist / tau)
