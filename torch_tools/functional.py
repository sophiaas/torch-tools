import torch


def l1_norm(x):
    return torch.sum(torch.abs(x))
