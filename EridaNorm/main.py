import torch
import torch.nn as nn

class EridaNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, beta_max=0.5, beta_init=-2.2):
        super().__init__()
        self.eps = eps
        self.beta_max = beta_max
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta_raw = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        x_f = x.float()
        mean = x_f.mean(dim=-1, keepdim=True)
        rms_sq = (x_f ** 2).mean(dim=-1, keepdim=True)
        beta_eff = torch.sigmoid(self.beta_raw) * self.beta_max
        denom = torch.sqrt(rms_sq + beta_eff * mean ** 2 + self.eps)
        out = (x_f / denom).to(x.dtype)
        return out * self.gamma
