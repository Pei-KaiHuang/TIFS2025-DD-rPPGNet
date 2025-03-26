import torch
import numpy as np
import torch.nn as nn

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()
        return

    def forward(self, x, y):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 1, keepdim = True)
        vy = y - torch.mean(y, dim = 1, keepdim = True)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost = 1 - r
        return cost
