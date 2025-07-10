import os
import shutil
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from utils.config import pretrain_config
from utils.utils import set_seed, get_logger_and_writer, create_model, ListApply, RecordTime, accuracy
from utils.get_dataset import get_pretrain_dataloader
from tqdm import tqdm
from copy import deepcopy


def run_step(config, inputs, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl):
    mixed_lamda = mml.get_lamda()
    rot_inputs = inputs[0].reshape([inputs[0].shape[0] * inputs[0].shape[1], *inputs[0].shape[2::]]).to(device)
    rot_labels = inputs[1].reshape(-1).long().to(device)
    device_labels = torch.tensor([[inputs[2][i] for _ in range(config["rot_classifier"]["num_classes"])] for i in range(len(inputs[2]))]).reshape(
        -1).long().to(device)
    mixed_inputs = inputs[0][:, 0, :, :].detach().clone().to(device)
    mixed_labels = inputs[2].detach().clone().long().to(device)

    loss_items = []
    acc_items = [0.0, 0.0, 0.0]

    rot_features = None
    for loss_name in config["mtl"]["item"]:
        if loss_name == "rot_cls":  # Phase Shift Prediction
            rot_features = encoder(rot_inputs) if rot_features is None else rot_features
            rot_preds = rot_classifier(rot_features)
            loss_items.append(cls(rot_preds, rot_labels))
            acc_items[0] = accuracy(rot_preds, rot_labels)
        elif loss_name == "sei_cls":
            rot_features = encoder(rot_inputs) if rot_features is None else rot_features
            device_preds = mixed_classifier(rot_features)
            loss_items.append(cls(device_preds, device_labels))
            acc_items[1] = accuracy(device_preds, device_labels)
        elif loss_name == "mml":  # Manifold Mixup
            mixed_features, mixed_labels_a, mixed_labels_b = encoder(mixed_inputs, mixed_labels, mixed_lamda)
            mixed_preds = mixed_classifier(mixed_features)
            loss_items.append(mml(mixed_preds, mixed_labels_a, mixed_labels_b))
            acc_items[2] = mixed_lamda * accuracy(mixed_preds, mixed_labels_a) + (1 - mixed_lamda) * accuracy(mixed_preds, mixed_labels_b)

    if len(loss_items) == 1:
        loss_items.append(loss_items[0] * 1.0)
    else:
        loss_items.append(mtl(*loss_items))  # Multi Tasks Using MTLoss
    # loss_items.append(loss_items[0] * 0.5 + loss_items[1] * 0.5 + loss_items[2])

    return loss_items, acc_items


def train(logger, writer, config, epoch, train_dataloader, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl, optimizers, schedulers):
    encoder.train()
    rot_classifier.train()
    mixed_classifier.train()
    acc_sum = [0.0, 0.0, 0.0]
    loss_sum = [0] * (config["mtl"]["num"] + 1)
    logger.info(f"==> lr = {optimizers[0].param_groups[0]['lr']}")
    for inputs in tqdm(train_dataloader, desc=f'Epoch: {epoch + 1}/{config["epoch"]}'):
        loss_items, acc_items = run_step(config, inputs, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl)
        optimizers.zero_grad()
        loss_items[-1].backward()
        optimizers.step()
        for idx, item in enumerate(acc_items):
            acc_sum[idx] += item
        for idx, item in enumerate(loss_items):
            loss_sum[idx] += item.item()
    schedulers.step()
    acc_sum = [v / len(train_dataloader) for v in acc_sum]
    loss_sum = [v / len(train_dataloader) for v in loss_sum]
    loss_sum_name = [n.upper() for n in config["mtl"]["item"]] + ["Total"]

    info_str = "==> Train Set: Rot-Acc: {:.2f}%, SEI-Acc: {:.2f}%, Mixed-Acc: {:.2f}%, " + ": {:.8f}, ".join(loss_sum_name) + ": {:8f}"
    logger.info(info_str.format(*acc_sum, *loss_sum))

    writer.add_scalar("Train Set - Rot Acc", acc_sum[0], epoch)
    writer.add_scalar("Train Set - SEI Acc", acc_sum[1], epoch)
    writer.add_scalar("Train Set - Mixed Acc", acc_sum[2], epoch)
    for idx, loss_name in enumerate(loss_sum_name):
        writer.add_scalar(f"Train Set - {loss_name} Loss", loss_sum[idx], epoch)


def val(logger, writer, config, epoch, val_dataloader, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl):
    encoder.eval()
    rot_classifier.eval()
    mixed_classifier.eval()
    acc_sum = [0.0, 0.0, 0.0]
    loss_sum = [0] * (config["mtl"]["num"] + 1)
    for inputs in tqdm(val_dataloader, desc=f'Epoch: {epoch + 1}/{config["epoch"]}'):
        with torch.no_grad():
            loss_items, acc_items = run_step(config, inputs, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl)
            for idx, item in enumerate(acc_items):
                acc_sum[idx] += item
            for idx, item in enumerate(loss_items):
                loss_sum[idx] += item.item()

    acc_sum = [v / len(val_dataloader) for v in acc_sum]
    loss_sum = [v / len(val_dataloader) for v in loss_sum]
    loss_sum_name = [n.upper() for n in config["mtl"]["item"]] + ["Total"]

    info_str = "==> Val Set: Rot-Acc: {:.2f}%, SEI-Acc: {:.2f}%, Mixed-Acc: {:.2f}%, " + ": {:.8f}, ".join(loss_sum_name) + ": {:8f}"
    logger.info(info_str.format(*acc_sum, *loss_sum))

    writer.add_scalar("Val Set - Rot Acc", acc_sum[0], epoch)
    writer.add_scalar("Val Set - SEI Acc", acc_sum[1], epoch)
    writer.add_scalar("Val Set - Mixed Acc", acc_sum[2], epoch)
    for idx, loss_name in enumerate(loss_sum_name):
        writer.add_scalar(f"Val Set - {loss_name} Loss", loss_sum[idx], epoch)

    return acc_sum, loss_sum


def train_and_val(record_time, logger, writer, config, train_dl, val_dl, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl, opts, schs,
                  checkpoint):
    if checkpoint is not None and "best_record" in checkpoint:
        best_record = checkpoint["best_record"]
    else:
        best_record = {
            "epoch": 0,
            "acc": [0.0, 0.0, 0.0],
            "loss": [0] * (config["mtl"]["num"] + 1)
        }
    for epoch in range(config["start_epoch"], config["end_epoch"]):
        logger.info("--------------------------------------------")
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}")
        record_time.start()
        train(logger, writer, config, epoch, train_dl, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl, opts, schs)
        acc, loss = val(logger, writer, config, epoch, val_dl, device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl)
        if epoch == config["start_epoch"] or (epoch > config["start_epoch"] and sum(best_record["loss"][0:-1]) > sum(loss[0:-1])):
            best_record["epoch"] = epoch
            best_record["acc"] = acc
            best_record["loss"] = loss
            torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "best_encoder.pth"))
            logger.info(f"==> Best model is saved in epoch {epoch + 1}.")
        if epoch % config["save_freq"] == 0 or epoch == config["epoch"] - 1:
            torch.save({
                "epoch": epoch,
                "config": config,
                "best_record": best_record,
                "encoder": encoder.state_dict(),
                "rot_classifier": rot_classifier.state_dict(),
                "mixed_classifier": mixed_classifier.state_dict(),
                "mtl": mtl.state_dict(),
                "optimizers": opts.state_dict(),
                "schedulers": schs.state_dict()
            }, os.path.join(config["exp_path"], "checkpoint.pth"))
        torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "final_encoder.pth"))
        logger.info("==> Time spend (current/mean/total/remain): {}/{}/{}/{}".format(*record_time.step()))
    shutil.copy(os.path.join(config["exp_path"], "best_encoder.pth"),
                os.path.join(config["save_path"], "best_encoder.pth"))
    shutil.copy(os.path.join(config["exp_path"], "final_encoder.pth"),
                os.path.join(config["save_path"], "final_encoder.pth"))
    logger.info("--------------------------------------------")
    logger.info(f"End. Best Record: {best_record}")


def pretext(config=None):
    config = pretrain_config() if config is None else config
    config["exp_type"] = config["exp_type"] + "_random_rot"
    logger, writer, exp_path, save_path = get_logger_and_writer(os.path.join("runs", config["exp_type"], config["exp_name"]))

    checkpoint = None
    if config["resume"] and isinstance(config["resume"], str):
        #  如果config["resume"]是checkpoint文件路径，则从config["resume"]恢复，如果不是，则拼接"checkpoint.pth"
        resume_path = os.path.join(save_path, config['resume'])
        resume_path = resume_path if os.path.isfile(resume_path) else os.path.join(resume_path, "checkpoint.pth")
        logger.info(f"==> Resume from {resume_path}")
        checkpoint = torch.load(resume_path)
        checkpoint["config"]["resume"] = config["resume"]
        config = checkpoint["config"]
        config["start_epoch"] = checkpoint["epoch"]

    set_seed(config["random_seed"])

    logger.info(f"==> Running file: {os.path.abspath(__file__)}")
    config["exp_path"] = exp_path
    config["save_path"] = save_path
    logger.info(f"==> Config: {config}")

    logger.info(f"==> Loading pretrain datasets.")
    train_dataloader, val_dataloader = get_pretrain_dataloader(config)

    logger.info(f"==> Instantiating specified model.")
    device = config["device"]

    conf_en = config["encoder"]
    if "TSLA" in conf_en["root"]:
        encoder = create_model(conf_en["root"], feature_dim=conf_en["feature_dim"], dtype=config["dataset"]["type"], **conf_en["TSLA_config"])
    else:
        encoder = create_model(conf_en["root"], feature_dim=conf_en["feature_dim"], dtype=config["dataset"]["type"])
    if torch.cuda.device_count() > 1:
        logger.info(f"==> Enable multi-GPU ({os.environ['CUDA_VISIBLE_DEVICES']}) parallel acceleration.")
        encoder = torch.nn.DataParallel(encoder, device_ids=range(torch.cuda.device_count()))
    encoder = encoder.to(device)
    if checkpoint is not None and "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])

    conf_cl = config["rot_classifier"]
    rot_classifier = create_model(conf_cl["root"], in_dim=conf_cl["in_dim"], num_classes=conf_cl["num_classes"])
    rot_classifier = rot_classifier.to(device)
    if checkpoint is not None and "rot_classifier" in checkpoint:
        rot_classifier.load_state_dict(checkpoint["rot_classifier"])
    conf_cl = config["mixed_classifier"]
    mixed_classifier = create_model(conf_cl["root"], in_dim=conf_cl["in_dim"], num_classes=conf_cl["num_classes"])
    mixed_classifier = mixed_classifier.to(device)
    if checkpoint is not None and "mixed_classifier" in checkpoint:
        mixed_classifier.load_state_dict(checkpoint["mixed_classifier"])

    logger.info(f"==> Creating loss functions.")
    cls = torch.nn.CrossEntropyLoss()
    mml = create_model(config["mml"]["root"], beta=config["mml"]["beta"])
    mtl = create_model(config["mtl"]["root"], num=config["mtl"]["num"]).to(device)
    if checkpoint is not None and "mtl" in checkpoint:
        mtl.load_state_dict(checkpoint["mtl"])

    logger.info(f"==> Generating optimizers for step 1.")
    conf_opt = config["optimizer"]
    optimizers = ListApply([AdamW(encoder.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                            AdamW(rot_classifier.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                            AdamW(mixed_classifier.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                            AdamW(mtl.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"])])
    if checkpoint is not None and "optimizers" in checkpoint:
        optimizers.load_state_dict(checkpoint["optimizers"])

    schedulers = ListApply([StepLR(op, step_size=conf_opt["step_size"], gamma=conf_opt["gamma"]) for op in optimizers])
    if checkpoint is not None and "schedulers" in checkpoint:
        schedulers.load_state_dict(checkpoint["schedulers"])

    logger.info(f"==> Starting pretext task.")
    step2_enable = False
    record_time = RecordTime(config["epoch"] - config["start_epoch"])
    if config["threshold"] > config["start_epoch"]:  # Only Pretrain Rot Prediction Ability
        fake_config = deepcopy(config)
        fake_config["end_epoch"] = config["threshold"]
        fake_config["mtl"]["item"] = ["rot_cls"]
        fake_config["mtl"]["num"] = 1
        del optimizers[-2:]
        del schedulers[-2:]
        train_and_val(record_time, logger, writer, fake_config, train_dataloader, val_dataloader, device, encoder, rot_classifier, mixed_classifier,
                      cls, mml, mtl, optimizers, schedulers, checkpoint)
        checkpoint = None
        # checkpoint文件更名为checkpoint_step1.pth
        os.rename(os.path.join(config["exp_path"], "checkpoint.pth"), os.path.join(config["exp_path"], "checkpoint_step1.pth"))
        logger.info(f"==> Generating new optimizers for step 2.")
        conf_opt = config["optimizer"]
        optimizers = ListApply([AdamW(encoder.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                                AdamW(rot_classifier.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                                AdamW(mixed_classifier.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                                AdamW(mtl.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"])])
        schedulers = ListApply([StepLR(op, step_size=conf_opt["step_size"], gamma=conf_opt["gamma"]) for op in optimizers])
        step2_enable = True
    if config["threshold"] <= config["start_epoch"] < config["epoch"] or step2_enable:  # Pretrain All Abilities
        fake_config = deepcopy(config)
        fake_config["start_epoch"] = config["threshold"]
        fake_config["end_epoch"] = config["epoch"]
        train_and_val(record_time, logger, writer, fake_config, train_dataloader, val_dataloader, device, encoder, rot_classifier, mixed_classifier,
                      cls, mml, mtl, optimizers, schedulers, checkpoint)
        # checkpoint文件更名为checkpoint_step1.pth
        os.rename(os.path.join(config["exp_path"], "checkpoint.pth"), os.path.join(config["exp_path"], "checkpoint_step2.pth"))
    writer.close()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # CUDA_VISIBLE_DEVICES=0 nohup python pretext.py -e CVTSLANet > pt_CVTSLANet.log 2>&1 &
    pretext(pretrain_config(encoder_name="CVTSLANet", classifiar_name="Linear", dataset_name="ads-b", input_type="iq", rot_num=8, feature_dim=1024, epoch_threshold=0))
