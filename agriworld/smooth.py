import torch
import torch.nn.functional as F


def smooth_relu(x, beta=10.0):
    return F.softplus(x, beta=beta)


def smooth_positive(x, beta=5.0):
    return smooth_relu(x, beta=beta)


def smooth_min(a, b, tau=0.1):
    return -tau * torch.logsumexp(torch.stack([-a / tau, -b / tau], dim=0), dim=0)


def smooth_clamp01(x, k=8.0):
    return torch.sigmoid(k * (x - 0.5)) * 1.1 - 0.05