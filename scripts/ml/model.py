"""
Graph neural network that predicts the fine-resolution rate of a target
element from the coarse multi-element detection scan.

  coarse nodes --[L x edge-conditioned message passing]--> node embeddings
  fine pixel   --[attention over its 4 surrounding nodes]-> rate

The message passing uses continuous edge kernels (a small MLP of the relative
offset gates each message), so the same layers run on irregular node sets -
nothing assumes a pixel grid. The prediction is anchored on the bilinear
interpolation of the coarse target map,

    rate = scale * ( base * exp(g) + floor(a) * mean(base) ),

with g bounded by a tanh and floor(a) = relu(softplus(a) - softplus(a0)), so an
untrained network IS the non-learned baseline exactly (g = 0, floor = 0) and
learning can only move away from it where the data say so. The additive term
lets it predict signal where the coarse scan read zero.
Trained with the Poisson negative log-likelihood of the fine COUNTS, which is
the correct loss for a noisy photon-counting label.
"""

import torch
from torch import nn


A0 = -4.0          # initial bias of the additive term (its value is subtracted)


class EdgeConv(nn.Module):
    """h_i <- norm( h_i + MLP([h_i, mean_j gate(offset_ij) * W h_j]) )"""

    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Linear(hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, src, dst, rel, deg):
        m = self.msg(h)[src] * torch.sigmoid(self.gate(rel))
        agg = torch.zeros_like(h).index_add_(0, dst, m) / deg
        return self.norm(h + self.upd(torch.cat([h, agg], dim=1)))


class RateGNN(nn.Module):
    def __init__(self, n_features, hidden=32, layers=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(EdgeConv(hidden) for _ in range(layers))
        self.att = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.val = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))
        nn.init.zeros_(self.head[-1].weight)          # start at the baseline
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0, A0]))
        self.floor0 = float(nn.functional.softplus(torch.tensor(A0)))

    def encode(self, x, src, dst, erel):
        """Node embeddings of the coarse graph."""
        deg = torch.zeros(x.shape[0], 1, device=x.device).index_add_(
            0, dst, torch.ones(len(dst), 1, device=x.device)).clamp_(min=1.0)
        h = self.enc(x)
        for conv in self.convs:
            h = conv(h, src, dst, erel, deg)
        return h

    def decode(self, h, nbr, qrel, base, rho, base_mean):
        """Relative rate (counts per coarse-count unit) at the queried pixels.
        nbr (Q, 4) node indices, qrel (Q, 4, 2) offsets, base (Q,) bilinear
        baseline, rho = fine / coarse pixel size."""
        hj = torch.cat([h[nbr], qrel], dim=-1)                     # (Q, 4, H+2)
        w = torch.softmax(self.att(hj), dim=1)
        z = (w * self.val(hj)).sum(dim=1)
        q = torch.stack([torch.log1p(base), torch.full_like(base, float(rho))], dim=1)
        g, a = self.head(torch.cat([z, q], dim=1)).unbind(dim=1)
        floor = torch.relu(nn.functional.softplus(a) - self.floor0)
        return base * torch.exp(4.0 * torch.tanh(g / 4.0)) + floor * base_mean


def poisson_nll(rate, counts, t=1.0):
    """Mean Poisson negative log-likelihood of `counts` observed for a time t
    at `rate` (constant terms dropped)."""
    lam = (rate * t).clamp(min=1e-6)
    return (lam - counts * torch.log(lam)).mean()
