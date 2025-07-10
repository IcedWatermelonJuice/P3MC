# 3 TSLANet Layer + KAN (GRAMKAN) + Complex Conv (PatchEmbed Only) + Non-Constant ASB Threshold
import torch
from torch import nn
import math
import warnings
import sys
import os
if not (sys.path[0] == '.' or sys.path[0] == 'models' or sys.path[0] == os.path.abspath('models')):
    from pathlib import Path
    sys.path.append(str(Path(__file__).absolute().parent))
from models.GRAMKAN import GRAMLayer as KANLinear


# copy from timm.models.layers.trunc_normal_
def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# copy from timm.models.layers.DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def _drop_path(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self._drop_path(x)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        if in_channels % 2 != 0 or out_channels % 2 != 0:
            raise ValueError("The number of input and output channels must be even.")
        in_channels, out_channels = in_channels // 2, out_channels // 2

        # Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def forward(self, x):  # shape of x : [batch, channel, axis1]
        x = x.float()
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        outputs = torch.cat((real, imaginary), dim=1)
        return outputs


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


# Conv1d --> ComplexConv
class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = ComplexConv(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, num_patch):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal(self.complex_weight_high, std=.02)
        trunc_normal(self.complex_weight, std=.02)
        threshold_length = int((num_patch + 1) / 2)
        self.threshold_param = nn.Parameter(torch.rand(threshold_length))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in, enable_adaptive_filter=True):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if enable_adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, num_patch, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, num_patch)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, enable_ICB=True, enable_ASB=True):
        # Check if both ASB and ICB are true
        if enable_ICB and enable_ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif enable_ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif enable_ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim, depth, num_classes, dropout_rate):
        super().__init__()
        self.depth = depth
        self.patch_embed = PatchEmbed(seq_len=seq_len, patch_size=patch_size, in_chans=num_channels, embed_dim=emb_dim)
        num_patch = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patch, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.input_layer = KANLinear(patch_size, emb_dim)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([TSLANet_layer(dim=emb_dim, num_patch=num_patch, drop=dropout_rate, drop_path=dpr[i]) for i in range(depth)])

        mlp_dim = emb_dim * 2
        # Classifier head
        self.head = nn.Sequential(
            KANLinear(emb_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            KANLinear(mlp_dim, num_classes)
        )

        # init weights
        trunc_normal(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cutmix_data(x, y, lam):
        '''Compute the cutmix data. Return mixed inputs, pairs of targets, and lambda'''
        # x.shape = (batch_size, token_num, dim_len)
        batch_size, _, x_len = x.shape
        cut_len = int(x_len * lam)
        start_index = torch.randint(0, x_len - cut_len, (1,)).item()
        end_index = start_index + cut_len
        batch_index = torch.randperm(batch_size).to(x.device)
        mixed_x = x.clone()
        mixed_x[:, :, start_index:end_index] = x[batch_index, :, start_index:end_index].clone()
        y_a, y_b = y, y[batch_index]

        return mixed_x, y_a, y_b

    def _set_forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        x = x.mean(1)
        x = self.head(x)
        return x

    def _set_forward_by_mixed_data(self, x, y, mixed_lambda):
        mixup_layer = 1
        y_a, y_b = None, None
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if mixup_layer == 0:
            x, y_a, y_b = self.cutmix_data(x, y, mixed_lambda)
        for tsla_blk_id, tsla_blk in enumerate(self.tsla_blocks):
            if tsla_blk_id == (mixup_layer - 1):
                x, y_a, y_b = self.cutmix_data(x, y, mixed_lambda)
            x = tsla_blk(x)
        x = x.mean(1)
        x = self.head(x)
        return x, y_a, y_b

    def forward(self, x, y=None, mixed_lambda=None):
        if y is not None or mixed_lambda is not None:
            assert mixed_lambda is not None, "Mixed lambda is required for CutMix"
            assert y is not None, "Label is required for CutMix"
            return self._set_forward_by_mixed_data(x, y, mixed_lambda)
        else:
            return self._set_forward(x)


def create_model(feature_dim=1024, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("TSLANet is only used for 'dtype = iq'.")
    return TSLANet(seq_len=kwargs["seq_len"], patch_size=kwargs["patch_size"], num_channels=kwargs["num_channels"], emb_dim=kwargs["emb_dim"],
                   depth=kwargs["depth"], num_classes=feature_dim, dropout_rate=kwargs["dropout_rate"])
    # return TSLANet(seq_len=4800, patch_size=32, num_channels=2, emb_dim=256, depth=1, num_classes=feature_dim, dropout_rate=0.3)


if __name__ == "__main__":
    model = TSLANet(4800, 32, 2, 512, 1, 10, 0.1)
    inputs = torch.randn(32, 2, 4800)
    outputs = model(inputs)
    print(outputs.shape)
