from utils.config import finetune_config
from downstream import downstream as downstream_dl
from downstream_lr import downstream as downstream_lr
from downstream_knn import downstream as downstream_knn
from downstream_svm import downstream as downstream_svm
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # CUDA_VISIBLE_DEVICES=1 nohup python downstream.py -e CVTSLANet -c lr > ft_CVTSLANet_lr.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup python downstream.py -e CVTSLANet -c knn > ft_CVTSLANet_knn.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=3 nohup python downstream.py -e CVTSLANet -c svm > ft_CVTSLANet_svm.log 2>&1 &
    from copy import deepcopy

    config = {
        "shot": [1, 5, 10, 15, 20],
        "max_iteration": 100,
        "max_epoch": 100,
        "encoder_name": "CVTSLANet",
        "dataset_name": "ads-b",
        "input_type": "iq",
        "feature_dim": 1024,
        "snr": [0, 5, 10, 15, 20, 25, 30],
        "snr_enable": False,
    }
    def_config = finetune_config(**config)
    def_config["exp_type"] = "Downstream_random_rot"
    def_config["encoder"]["pretrain_path"] = def_config["encoder"]["pretrain_path"].replace("./runs/Pretext", "./runs/Pretext_random_rot")
    shot = def_config["dataset"]["shot"]
    snr = def_config["dataset"]["snr"]

    if def_config["classifier"]["name"] == "lr":
        def_config["exp_name"] = def_config["exp_name"].replace(def_config["encoder"]["name"], def_config["encoder"]["name"] + "_LR")
        downstream = downstream_lr
    elif def_config["classifier"]["name"] == "knn":
        def_config["exp_name"] = def_config["exp_name"].replace(def_config["encoder"]["name"], def_config["encoder"]["name"] + "_KNN")
        downstream = downstream_knn
    elif def_config["classifier"]["name"] == "svm":
        def_config["exp_name"] = def_config["exp_name"].replace(def_config["encoder"]["name"], def_config["encoder"]["name"] + "_SVM")
        downstream = downstream_svm
    else:
        downstream = downstream_dl

    for n in snr:
        for s in shot:
            ft_config = deepcopy(def_config)
            ft_config["dataset"]["shot"] = s
            ft_config["dataset"]["snr"] = n
            downstream(ft_config)