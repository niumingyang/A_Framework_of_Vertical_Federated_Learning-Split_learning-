import torch
from utils.metric import compute_auc


def emb_attack(embedding, target):
    # 对mini-batch内的embedding,计算其经验均值并做归一化处理
    mean_emb = torch.mean(embedding, dim=0)
    mean_reduced_emb = embedding - mean_emb

    # 对归一化后的embedding做奇异值分解,取最大奇异值对应的奇异向量v
    u, s, v = torch.svd(mean_reduced_emb)
    top_singular_vector = v[:, 0]

    # 做矩阵乘法将embedding分成两个簇
    pred = torch.mv(mean_reduced_emb, top_singular_vector)

    # 计算auc
    auc = compute_auc(target, pred)

    return max(auc, 1-auc)
