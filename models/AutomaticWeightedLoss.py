'''
This file is from https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
'''
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params£º
        num: int£¬the number of loss
        x: multi-task loss
    Examples£º
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    # 修改原AWL类，增加loss weight归一化过程与最小阈值，防止loss爆炸/消失
    def normalize_params(self, threshold=1e-3):
        with torch.no_grad():
            max_val = torch.max(self.params)
            self.params.data = self.params.data / max_val
            self.params.data[self.params.data < threshold] = threshold

    def forward(self, *x):
        self.normalize_params()
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def create_model(num=3):
    return AutomaticWeightedLoss(num=num)


if __name__ == '__main__':
    awl = create_model(2)
    print(awl.parameters())
