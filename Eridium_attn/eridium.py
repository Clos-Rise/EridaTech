import torch
import torch.nn as nn
import torch.nn.functional as F


class EridiumAttention(nn.Module):
    def __init__(self, dim, num_heads=8, eps=1e-4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.zeros(num_heads))
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x, causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        temp = F.softplus(self.temp).view(1, self.num_heads, 1, 1)
        scores = scores / (temp + self.eps)
        if causal:
            mask = torch.arange(N, device=x.device)[:, None] < torch.arange(N, device=x.device)[None, :]
            scores = scores.masked_fill(mask[None, None, ...], -1e4)
        pos = scores + torch.sqrt(scores * scores + self.eps)
        weights = pos / (pos.sum(dim=-1, keepdim=True) + self.eps)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out(out)


class EridiumFast(nn.Module):
    def __init__(self, dim, num_heads=8, eps=1e-4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.zeros(num_heads))
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x, causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        temp = F.softplus(self.temp).view(1, self.num_heads, 1, 1)
        scores = scores / (temp + self.eps)
        if causal:
            mask = torch.arange(N, device=x.device)[:, None] < torch.arange(N, device=x.device)[None, :]
            scores = scores.masked_fill(mask[None, None, ...], -1e4)
        pos = F.relu(scores) + self.eps
        weights = pos / (pos.sum(dim=-1, keepdim=True) + self.eps)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out(out)
