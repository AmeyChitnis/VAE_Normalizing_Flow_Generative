import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, swap=False):
        super().__init__()
        self.swap = swap
        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        if self.swap:
            x2, x1 = x.chunk(2, dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([x1, z2], dim=1) if not self.swap else torch.cat([z2, x1], dim=1)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        if self.swap:
            z2, z1 = z.chunk(2, dim=1)
        else:
            z1, z2 = z.chunk(2, dim=1)
        h = self.net(z1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([z1, x2], dim=1) if not self.swap else torch.cat([x2, z1], dim=1)
        return x

class NormalizingFlowModel(nn.Module):
    def __init__(self, flows, input_dim):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.register_buffer("loc", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))

    def forward(self, x):
        log_det = 0
        for flow in self.flows:
            x, ld = flow(x)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z

    def log_prob(self, x):
        z, log_det = self.forward(x)
        base_dist = torch.distributions.Normal(self.loc.to(x.device), self.scale.to(x.device))
        return base_dist.log_prob(z).sum(dim=1) + log_det
