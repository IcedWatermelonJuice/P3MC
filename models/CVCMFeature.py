import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig


class ComplexConv(nn.Module):
    def __init__(self, in_channels=1 * 2, out_channels=8 * 2, kernel_size=50, stride=50, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        assert in_channels % 2 == 0, "The number of input channels must be even."
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        # Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shape of x : [batch, channel, axis1]
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        outputs = torch.cat((real, imaginary), dim=1)
        return outputs


class CV_Embedding(nn.Module):
    def __init__(self, patch_dim=8 * 2, patch_size=50):
        super().__init__()
        self.cvcnn = ComplexConv(1 * 2, patch_dim, kernel_size=patch_size, stride=patch_size)
        patch_num = int(4800 / patch_size)
        self.pos = nn.Parameter(
            torch.zeros(1, patch_num, patch_dim)
        )
        self.norm = nn.BatchNorm1d(patch_dim, eps=1e-6)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.cvcnn(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos
        x_shape = x.shape
        x = x.reshape(x_shape[0]* x_shape[1], x_shape[2])
        x = self.norm(x)
        x = x.reshape(x_shape)
        x = self.act(x)
        return x


class CVCM(nn.Module):
    def __init__(self, patch_dim=6 * 2, patch_size=50, emb_dim=256):
        super().__init__()
        self.patch_emb = CV_Embedding(patch_dim, patch_size)
        # self.mambaconfig = MambaConfig(d_model=patch_dim, n_layers=3, d_state=8, d_conv=3, expand_factor=64)
        self.mambaconfig = MambaConfig(d_model=patch_dim, n_layers=10, d_state=8, d_conv=3, expand_factor=64)
        self.mamba = Mamba(self.mambaconfig)
        self.fc = nn.Sequential(nn.Linear(patch_dim, emb_dim), nn.ReLU())

    @staticmethod
    def cutmix_data(x, y, lam):
        '''Compute the cutmix data. Return mixed inputs, pairs of targets, and lambda'''
        # batch_size = x.shape[0]
        # index = torch.randperm(batch_size).to(x.device)
        # mixed_x = lam * x + (1 - lam) * x[index, :]
        # y_a, y_b = y, y[index]
        batch_size, _, x_len = x.shape
        cut_len = int(x_len * lam)
        start_index = torch.randint(0, x_len - cut_len, (1,)).item()
        end_index = start_index + cut_len
        batch_index = torch.randperm(batch_size).to(x.device)
        mixed_x = x.clone()
        mixed_x[:, :, start_index:end_index] = x[batch_index, :, start_index:end_index].clone()
        y_a, y_b = y, y[batch_index]

        return mixed_x, y_a, y_b

    def forward(self, x, y=None, mixed_lambda=None):
        x = self.patch_emb(x)
        if y is not None or mixed_lambda is not None:
            assert mixed_lambda is not None, "Mixed lambda is required for CutMix"
            assert y is not None, "Label is required for CutMix"
            layer_block = nn.Sequential(*self.mamba.layers)
            mixed_layer = torch.randint(0, len(layer_block) + 1, (1,)).item()
            x = layer_block[:mixed_layer](x)
            x, y_a, y_b = self.cutmix_data(x, y, mixed_lambda)
            x = layer_block[mixed_layer:](x)
            x = x.mean(dim=1)
            x = self.fc(x)
            return x, y_a, y_b
        else:
            x = self.mamba(x)
            x = x.mean(dim=1)
            x = self.fc(x)
            return x

def create_model(feature_dim=4096, dtype="iq"):
    if "iq" not in dtype:
        raise ValueError("CVCM is only used for 'dtype = iq'.")
    return CVCM(emb_dim=feature_dim)


if __name__ == "__main__":
    x = torch.rand(16, 2, 4800)
    print(x.shape)
    model = CVCM()
    features, pred_y = model(x)
    print(features.shape, pred_y.shape)
