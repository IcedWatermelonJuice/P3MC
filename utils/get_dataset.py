import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def rot_data(x, rot_num=4):
    """
    信号x分割为rot_num个小段，小段进行相位偏移
    :param x: 待处理信号x
    :param rot_num: 平均分割的需要相位偏移的小段数量
    :return: 相位偏移后的信号x_rot, 相位偏移后的标签y_rot
    """
    # 尽可能平均分割的小段的index
    indexs = np.array_split(np.arange(x.shape[2]), bps_num)
    x_rot = []
    y_rot = []
    # 相位偏移的旋转矩阵
    # 生成一个随机的旋转角度，范围1-90度
    radians = np.deg2rad(np.random.randint(1, 91, size=(bps_num, x.shape[0])))
    cos = np.cos(radians)
    sin = np.sin(radians)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]]).transpose(2, 3, 0, 1)
    # 对每个小段进行旋转
    for label, index in enumerate(indexs):
        x_proposed = np.array(x, copy=True)
        x_proposed[:, :, index] = np.matmul(rotation_matrix[label], x_proposed[:, :, index])
        x_rot.append(x_proposed)
        y_rot.append(np.ones(x.shape[0]) * label)
    return np.stack(x_rot, axis=1), np.stack(y_rot, axis=1).astype(np.uint8)
    

def add_noise(x, snr=20):
    """
        为信号添加噪声
        :param x: 待处理信号x, 形状为 batch_size, 2, signal_length
        :param snr: 信噪比
        :return: 添加噪声后的信号x_noisy
        """
    # 获取信号长度
    signal_length = x.shape[2]
    # 计算信号的功率
    signal_power = np.sum(np.power(x, 2), axis=2) / signal_length
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr / 10))
    # 生成与输入数组形状相同的高斯噪声
    noise = np.random.normal(size=x.shape)
    # 计算当前噪声功率
    current_noise_power = np.sum(np.power(noise, 2), axis=2) / signal_length
    # 根据噪声功率对噪声进行缩放，确保awgn的功率是符合要求的
    noise = noise * np.sqrt(noise_power / current_noise_power)[..., np.newaxis]
    # 将噪声添加到输入数组
    x_noisy = x + noise
    return x_noisy


def default_normalize_fn(x):
    return x


def sample_max_min(x):
    max_value = x.max(axis=2)
    min_value = x.min(axis=2)
    max_value = np.expand_dims(max_value, 2)
    min_value = np.expand_dims(min_value, 2)
    max_value = max_value.repeat(x.shape[2], axis=2)
    min_value = min_value.repeat(x.shape[2], axis=2)
    x = (x - min_value) / (max_value - min_value)
    return x


def entire_max_min(x):
    max_value = x.max()
    min_value = x.min()
    x = (x - min_value) / (max_value - min_value)
    return x


def power_normalize_fn(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)).max()
        x[i] = x[i] / np.power(max_power, 1 / 2)
    return x


def load_data(dataset_root, num_class, suffix, ch_type=None):
    ch_type = f"_{ch_type}" if ch_type else ""
    x = np.load(os.path.expanduser(os.path.join(dataset_root, f"X_{suffix}_{num_class}Class{ch_type}.npy")))
    y = np.load(os.path.expanduser(os.path.join(dataset_root, f"Y_{suffix}_{num_class}Class.npy")))

    if len(x.shape) == 3 and x.shape[1] != 2:
        x = x.transpose((0, 2, 1))

    return x[:, :, :4800], y


def pt_train_data(dataset_root, num_class, rot_num, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "train")
    if "/stft" in dataset_root:
        y_rot = np.array([[0, 1, 2, 3] for _ in range(len(x))]).astype(np.uint8)
    else:
        x, y_rot = rot_data(x, rot_num)
        shape = x.shape
        x = x.reshape((shape[0] * shape[1], shape[2], shape[3]))
        x = normalize_dataX(x)
        x = x.reshape(*shape)
    y_device = y.astype(np.uint8)

    return x, y_rot, y_device


def ft_train_data(random_seed, dataset_root, num_class, k_shot, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "train")
    if len(x.shape) == 5:
        x = x[:, 0, :, :, :]

    train_index_shot = []
    random.seed(random_seed)
    for i in range(num_class):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += random.sample(index_classi, k_shot)
    x = x[train_index_shot]
    y = y[train_index_shot]

    x = normalize_dataX(x)
    y = y.astype(np.uint8)
    return x, y


def ft_test_data(dataset_root, num_class, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "test")
    if len(x.shape) == 5:
        x = x[:, 0, :, :, :]

    x = normalize_dataX(x)
    y = y.astype(np.uint8)

    return x, y


def get_pretrain_dataloader(opt):
    opt_dataset = opt["dataset"]
    rot_num = opt["rot_classifier"]["num_classes"]

    if opt_dataset["normalize"] == "sample":
        normalize_fn = sample_max_min
    elif opt_dataset["normalize"] == "dataset":
        normalize_fn = entire_max_min
    elif opt_dataset["normalize"] == "power":
        normalize_fn = power_normalize_fn
    else:
        normalize_fn = default_normalize_fn

    X_train, Y_rot_train, Y_device_train = pt_train_data(opt_dataset["root"], opt_dataset["num_classes"], rot_num, normalize_fn)
    X_train, X_val, Y_rot_train, Y_rot_val, Y_device_train, Y_device_val = train_test_split(X_train, Y_rot_train, Y_device_train,
                                                                                            test_size=opt_dataset["ratio"],
                                                                                            random_state=opt["random_seed"])

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_rot_train), torch.tensor(Y_device_train))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_rot_val), torch.tensor(Y_device_val))

    train_dataloader = DataLoader(train_dataset, batch_size=opt_dataset['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt_dataset['batch_size'], shuffle=True)
    return train_dataloader, val_dataloader


def get_finetune_dataloader(opt):
    opt_dataset = opt["dataset"]

    if opt_dataset["normalize"] == "sample":
        normalize_fn = sample_max_min
    elif opt_dataset["normalize"] == "dataset":
        normalize_fn = entire_max_min
    elif opt_dataset["normalize"] == "power":
        normalize_fn = power_normalize_fn
    else:
        normalize_fn = default_normalize_fn

    X_train, Y_train = ft_train_data(opt["random_seed"], opt_dataset["root"], opt_dataset["num_classes"], opt_dataset["shot"], normalize_fn)
    X_test, Y_test = ft_test_data(opt_dataset["root"], opt_dataset["num_classes"], normalize_fn)
    if opt_dataset["snr"] is not None: X_test = add_noise(X_test, snr=opt_dataset["snr"])

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test))

    train_dataloader = DataLoader(train_dataset, batch_size=opt_dataset['train_batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt_dataset['test_batch_size'], shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    from config import pretrain_config

    pt_conf = pretrain_config()
    pt_train_dataloader, pt_val_dataloader = get_pretrain_dataloader(pt_conf)

    print("end")
