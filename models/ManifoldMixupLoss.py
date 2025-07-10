from torch import nn
import numpy as np


class ManifoldMixupLoss(nn.Module):
    def __init__(self, beta=2.0):
        super(ManifoldMixupLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.beta = beta
        self.lamda = None

    def get_lamda(self):
        self.lamda = np.random.beta(self.beta, self.beta)
        return self.lamda

    def forward(self, pred, y_a, y_b):
        assert self.lamda is not None, 'lamda is not initialized, please call get_lamda() first'
        return self.lamda * self.criterion(pred, y_a) + (1 - self.lamda) * self.criterion(pred, y_b)


def create_model(beta=2.0):
    return ManifoldMixupLoss(beta)
