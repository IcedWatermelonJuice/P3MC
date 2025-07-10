from torchvision.models.resnet import resnet18
from torch import nn
import torch


def resnet_init(module, in_ch=2):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3 and child.out_channels == 64:
            conv2d = nn.Conv2d(in_ch, child.out_channels, kernel_size=child.kernel_size[0], stride=child.stride[0],
                               padding=child.padding[0], dilation=child.dilation[0], groups=child.groups,
                               bias=child.bias is not None,
                               padding_mode=child.padding_mode)
            setattr(module, name, conv2d)
        else:
            resnet_init(child, in_ch=in_ch)
    return module


def resnet_init_1d(module, in_ch=2):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            conv1d = nn.Conv1d(in_ch if (child.in_channels == 3 and child.out_channels == 64) else child.in_channels,
                               child.out_channels, kernel_size=child.kernel_size[0], stride=child.stride[0],
                               padding=child.padding[0], dilation=child.dilation[0], groups=child.groups,
                               bias=child.bias is not None, padding_mode=child.padding_mode)
            setattr(module, name, conv1d)
        elif isinstance(child, nn.BatchNorm2d):
            bn1d = nn.BatchNorm1d(child.num_features, eps=child.eps, momentum=child.momentum, affine=child.affine,
                                  track_running_stats=child.track_running_stats)
            setattr(module, name, bn1d)
        elif isinstance(child, nn.MaxPool2d):
            maxpool1d = nn.MaxPool1d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding,
                                     dilation=child.dilation, return_indices=child.return_indices,
                                     ceil_mode=child.ceil_mode)
            setattr(module, name, maxpool1d)
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            avgpool1d = nn.AdaptiveAvgPool1d(output_size=child.output_size[0])
            setattr(module, name, avgpool1d)
        else:
            resnet_init_1d(child, in_ch=in_ch)
    return module


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


def custom_forward(self, x, y=None, mixed_lambda=None):
    if y is not None or mixed_lambda is not None:
        assert mixed_lambda is not None, "Mixed lambda is required for CutMix"
        assert y is not None, "Label is required for CutMix"
        layer_block = nn.Sequential(nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool), self.layer1, self.layer2, self.layer3, self.layer4)
        mixed_layer = torch.randint(0, len(layer_block) + 1, (1,)).item()
        x = layer_block[:mixed_layer](x)
        x, y_a, y_b = cutmix_data(x, y, mixed_lambda)
        x = layer_block[mixed_layer:](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, y_a, y_b
    else:
        return self._forward_impl(x)


def ResNet18Feature(feature_dim=4096, in_ch=2, dtype="stft"):
    model = resnet18(num_classes=feature_dim)
    model.fc = nn.Sequential(model.fc, nn.ReLU())
    model.forward = custom_forward.__get__(model, model.__class__)
    return resnet_init(model, in_ch) if "iq" not in dtype else resnet_init_1d(model, in_ch)


def create_model(feature_dim=4096, dtype="stft"):
    return ResNet18Feature(feature_dim, in_ch=2, dtype=dtype)


if __name__ == "__main__":
    import torch

    model = create_model(dtype="stft")
    print(model)
    x = torch.randn(32, 2, 128, 128).float()
    f = model(x)
    print(f.shape, f.dtype)
