import torch


def r2_loss(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return 1 - r2


def custom_loss_fn(pred, target):
    return torch.mean(0.6 * r2_loss(pred, target) + 0.4 * (pred - target) ** 2)
