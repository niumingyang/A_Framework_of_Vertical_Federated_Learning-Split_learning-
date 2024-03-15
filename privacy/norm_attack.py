import torch
from utils.metric import compute_auc


def norm_attack(grad, target):
    pred = torch.norm(grad, p=2, dim=1)
    return compute_auc(target, pred)
