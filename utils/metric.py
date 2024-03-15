import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_auc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    if np.unique(y_true).__len__() == 1:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_pred)
    return auc


def compute_auc_acc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    if np.unique(y_true).__len__() == 1:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_pred)
    y_pred = [1.0 if y > 0.5 else 0.0 for y in y_pred]
    acc = accuracy_score(y_true, y_pred)
    return auc, acc


def compute_prec_rec(y_true, y_pred, threshold):
    y_true = y_true.detach().cpu().numpy()
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    y_pred = y_pred.detach().cpu().numpy()
    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1.:
                TP_samples_num += 1
            else:
                TN_samples_num += 1
            right_samples_num += 1
        else:
            if y_pred[i] == 1.:
                FP_samples_num += 1
            else:
                FN_samples_num += 1
            wrong_samples_num += 1

    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0

    return precision, recall


def compute_f1(precision, recall):
    if (precision + recall) != 0:
        f1 = (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1


def precision_recall(output, target):
    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    y_true = np.array(target.clone().detach().cpu())
    y_pred = np.array(pred.clone().detach().cpu()[0])
    if sum(y_pred) == 0:
        y_pred = np.ones_like(y_pred)
    # print("y_true:", y_true)
    # print("y_pred:", y_pred)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1.:
                TP_samples_num += 1
            else:
                TN_samples_num += 1
            right_samples_num += 1
        else:
            if y_pred[i] == 1.:
                FP_samples_num += 1
            else:
                FN_samples_num += 1
            wrong_samples_num += 1

    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0

    return precision, recall
