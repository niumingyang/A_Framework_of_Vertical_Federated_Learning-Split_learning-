import torch
import torch.nn as nn

class DisCorLoss(nn.Module):
    def __init__(self):
        super(DisCorLoss, self).__init__()

    def dist_mat(self, x):
        y = x[:, None] - x
        if len(y.shape)==3:
            return torch.norm(y, p=2, dim=2)
        else:
            return torch.abs(y)

    def normalize(self, x):
        d = self.dist_mat(x)
        dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean() 
        return dn

    def distance_correlation(self, x, y):
        A = self.normalize(x)
        B = self.normalize(y)
    
        dc = (A * B).mean()
        dvx = (A ** 2).mean()
        dvy = (B ** 2).mean()
        temp = (dvx * dvy) ** 0.5
        if temp == 0:
            dcor = torch.tensor(1e-8)
        else:
            dcor = dc / temp
        return dcor

    def forward(self, pred, truth):
        dcor = self.distance_correlation(pred.float(), truth.float())
        return torch.log(dcor)