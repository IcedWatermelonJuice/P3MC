from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=4):
        super(LinearClassifier, self).__init__()
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.drop(x)
        return self.fc(x)


def create_model(in_dim=2048, num_classes=4):
    return LinearClassifier(in_dim, num_classes)
