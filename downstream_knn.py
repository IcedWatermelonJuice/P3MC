import os
import shutil

import numpy as np
import pandas as pd
import torch
from utils.utils import SummaryWriter
from utils.config import finetune_config
from utils.utils import set_seed, get_logger_and_writer, create_model
from utils.get_dataset import get_finetune_dataloader
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


def extract_features(encoder, dataloader, device, step="train"):
    encoder.eval()
    features = []
    targets = []
    for inputs, labels in tqdm(dataloader, desc=f"Extracting features from {step} dataset"):
        with torch.no_grad():
            inputs = inputs.to(device)
            features.append(encoder(inputs).cpu().numpy())
            targets.append(labels.numpy())
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    return features, targets


def run_step(logger, encoder, classifier, train_dataloader, test_dataloader, device):
    features, labels = extract_features(encoder, train_dataloader, device, step="train")
    classifier.fit(features, labels)
    features, labels = extract_features(encoder, test_dataloader, device, step="test")
    acc = classifier.score(features, labels)
    acc = acc * 100
    logger.info(f"Acc: {acc:.2f}%")
    return acc


def finetune(config, logger):
    set_seed(config["random_seed"])

    train_dataloader, test_dataloader = get_finetune_dataloader(config)

    device = config["device"]
    conf_en = config["encoder"]
    if "TSLA" in conf_en["root"]:
        encoder = create_model(conf_en["root"], feature_dim=conf_en["feature_dim"], dtype=config["dataset"]["type"], **conf_en["TSLA_config"])
    else:
        encoder = create_model(conf_en["root"], feature_dim=conf_en["feature_dim"], dtype=config["dataset"]["type"])
    if config["encoder"]["pretrain_path"]:
        encoder.load_state_dict(torch.load(config["encoder"]["pretrain_path"]))
    encoder = encoder.to(device)

    classifier = KNeighborsClassifier()

    acc = run_step(logger, encoder, classifier, train_dataloader, test_dataloader, device)

    return acc


def downstream(config=None):
    config = finetune_config() if config is None else config
    if config["dataset"]["snr"] is None:
        save_path = os.path.join("runs", config["exp_type"], config["exp_name"], f"{config['dataset']['shot']}Shot")
    else:
        save_path = os.path.join("runs", config["exp_type"], config["exp_name"], f"{config['dataset']['snr']}dB", f"{config['dataset']['shot']}Shot")
    logger, _, exp_path, save_path = get_logger_and_writer(save_path, create_writer=False)
    config["exp_path"] = exp_path
    config["save_path"] = os.path.dirname(save_path)
    logger.info(f"==> Running file: {os.path.abspath(__file__)}")
    logger.info(f"==> Config: {config}")
    acc_list = []
    for i in range(config['iteration']):
        writer = SummaryWriter(os.path.join(exp_path, "writer", str(i)))
        logger.info("--------------------------------------------")
        logger.info(f"Iteration {i + 1}/{config['iteration']}")
        config["random_seed"] = config["random_seed"] + i  # 2024 + 0 + 1 + 2 + 3 + ... + (config['iteration'] - 1)
        acc = finetune(config, logger)
        acc_list.append(acc)
        writer.close()
    df = pd.DataFrame(acc_list)
    df.to_excel(os.path.join(config["exp_path"], "test_result.xlsx"))
    shutil.copy(os.path.join(config["exp_path"], "test_result.xlsx"),
                os.path.join(config["save_path"], f"{config['exp_name']}_{config['dataset']['shot']}Shot.xlsx"))


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # CUDA_VISIBLE_DEVICES=2 nohup python downstream_knn.py -e CVTSLANet > ft_CVTSLANet_knn.log 2>&1 &
    from copy import deepcopy

    config = {
        "shot": [1, 5, 10, 15, 20],
        "max_iteration": 100,
        "max_epoch": 100,
        "encoder_name": "CVTSLANet",
        "dataset_name": "ads-b",
        "input_type": "iq",
        "feature_dim": 1024,
        "snr": [0, 10, 20, 30],
        "snr_enable": False,
    }
    def_config = finetune_config(**config)
    def_config["exp_name"] = def_config["exp_name"].replace(def_config["encoder"]["name"], def_config["encoder"]["name"] + "_KNN")
    shot = def_config["dataset"]["shot"]
    snr = def_config["dataset"]["snr"]
    for n in snr:
        for s in shot:
            ft_config = deepcopy(def_config)
            ft_config["dataset"]["shot"] = s
            ft_config["dataset"]["snr"] = n
            downstream(ft_config)
