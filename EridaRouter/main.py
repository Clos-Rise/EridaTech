import torch
import torch.nn as nn
import torch.nn.functional as F


class EridaRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2, balance_lambda=1.0, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_lambda = balance_lambda
        self.eps = eps
        self.gate_proj = nn.Linear(dim, num_experts, bias=False)
        self.temp = nn.Parameter(torch.zeros(1))
        self.sharp = nn.Parameter(torch.ones(1) * 10.0)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

    def forward(self, x):
        B, N, D = x.shape
        scores = self.gate_proj(x)
        temp = F.softplus(self.temp)
        scores = scores / (temp + self.eps)
        gate = scores + torch.sqrt(scores * scores + self.eps)
        topk_vals, topk_idx = torch.topk(gate, self.top_k, dim=-1)
        hard_mask = torch.zeros_like(gate).scatter_(-1, topk_idx, 1.0)
        kth = topk_vals[..., -1:].detach()
        sharp = F.softplus(self.sharp)
        diff = gate - kth
        soft_att = torch.sigmoid(diff * sharp)
        soft_mask = hard_mask + (1 - hard_mask) * soft_att
        gate_masked = gate * soft_mask
        weights = gate_masked / (gate_masked.sum(dim=-1, keepdim=True) + self.eps)
        usage = weights.mean(dim=[0, 1])
        balance_loss = self.balance_lambda * usage.var() * self.num_experts
        active = (weights > 0.01).float().mean(dim=[0, 1])
        return weights, topk_idx, balance_loss, active
