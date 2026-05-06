import torch
from torch.optim.optimizer import Optimizer

class Erida(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['s'] = torch.zeros_like(p)
                m = state['m']
                s = state['s']
                step = state['step'] + 1
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                s.mul_(beta2).add_(g.abs(), alpha=1 - beta2)
                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                m_hat = m / bias1
                s_hat = s / bias2
                update = m_hat / (s_hat + eps)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                state['step'] = step
        return loss
