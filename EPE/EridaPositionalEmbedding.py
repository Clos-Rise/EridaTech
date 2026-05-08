import torch
import torch.nn as nn
import torch.nn.functional as F


class EPE(nn.Module):
    def __init__(self, num_heads, tau_init=8.0, beta=50.0):
        super().__init__()
        self.num_heads = num_heads
        self.beta = beta
        self.slopes_raw = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.tau_raw = nn.Parameter(torch.ones(num_heads) * tau_init)

    @property
    def slopes(self):
        return F.softplus(self.slopes_raw, beta=self.beta)

    @property
    def tau(self):
        return F.softplus(self.tau_raw, beta=self.beta)

    def forward(self, seq_len_q, seq_len_k=None):
        if seq_len_k is None:
            seq_len_k = seq_len_q
        dist = torch.arange(seq_len_q)[:, None] - torch.arange(seq_len_k)[None, :]
        dist = dist.abs().float()
        slopes = self.slopes.view(-1, 1, 1)
        tau = self.tau.view(-1, 1, 1)
        return -slopes * dist / (1.0 + dist / tau)
