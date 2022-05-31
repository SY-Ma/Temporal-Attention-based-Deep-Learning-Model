# @Time    : 2021/12/15 20:43
# @Author  : SY.M
# @FileName: loss_function.py

import torch
from torch.nn import Module

class Classify_Loss(Module):
    def __init__(self):
        super(Classify_Loss, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, pre, label):
        loss = self.loss_func(pre, label.long())

        return loss


class Predict_Loss(Module):
    def __init__(self):
        super(Predict_Loss, self).__init__()

        # self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.loss_func = torch.nn.L1Loss(reduction='sum')
        # self.loss_func = torch.nn.SmoothL1Loss(reduction='sum')
        # self.loss_func = torch.nn.NLLLoss2d(reduction='mean')

    def forward(self, pre, y):

        y = y.unsqueeze(-1)
        loss = self.loss_func(pre.float(), y.float())

        return loss