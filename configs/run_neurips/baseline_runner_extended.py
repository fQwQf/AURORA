import sys
import os

# Args: METHOD DATASET ALPHA GPU NUM_ROUNDS LOCAL_EPOCHS
METHOD = sys.argv[1]
DATASET = sys.argv[2]
ALPHA = float(sys.argv[3])
GPU = sys.argv[4]
NUM_ROUNDS = int(sys.argv[5])
LOCAL_EPOCHS = int(sys.argv[6])

BENCH = "/data1/tongjizhou/AURORA_bench/IntactOFL/source"
sys.path.insert(0, BENCH)
os.chdir(BENCH)

from commona_libs import *
from dataset_helper import get_fl_dataset
from oneshot_algorithms import *
from models import get_model

DATA_ROOTS = {
    "CIFAR10": "/data1/tongjizhou/FAFI_ICML25/data/",
    "CIFAR100": "/data1/tongjizhou/FAFI_ICML25/data/",
    "SVHN": os.path.expanduser("~/datasets/"),
    "Tiny-ImageNet": os.path.expanduser("~/datasets/"),
}

NUM_CLASSES = {"CIFAR10": 10, "CIFAR100": 100, "SVHN": 10, "Tiny-ImageNet": 200}
CHANNELS = {"CIFAR10": 3, "CIFAR100": 3, "SVHN": 3, "Tiny-ImageNet": 3}

config = {
    "exp_name": f"{METHOD}_{DATASET}_a{ALPHA}_ext",
    "dataset": {"data_name": DATASET, "root_path": DATA_ROOTS[DATASET],
                "train_batch_size": 128, "test_batch_size": 256,
                "channels": CHANNELS[DATASET], "num_classes": NUM_CLASSES[DATASET],
                "image_size": 64 if DATASET == "Tiny-ImageNet" else 32},
    "distribution": {"type": "dirichlet", "label_num_per_client": 2, "alpha": ALPHA},
    "client": {"num_clients": 5},
    "server": {"num_rounds": NUM_ROUNDS, "frac_clients": 0.1, "lr": 0.01,
               "local_epochs": LOCAL_EPOCHS, "optimizer": "sgd", "momentum": 0.9,
               "weight_decay": 0.0001, "loss_name": "ce", "model_name": "resnet18",
               "aggregated_by_datasize": False, "lr_decay_per_round": 0.998},
    "device": f"cuda:{GPU}",
    "checkpoint": {"save_path": f"/data1/tongjizhou/AURORA_bench/IntactOFL/checkpoints/{METHOD}_{DATASET}_ext/", "save_freq": 10},
    "resume": False, "resume_best": False, "seed": 42,
    "dense": {"nz": 256, "ngf": 64, "g_steps": 30, "lr_g": 0.001, "synthesis_batch_size": 128,
              "batch_size": 128, "adv": 1.0, "bn": 1.0, "oh": 1.0, "his": True,
              "batchonly": False, "batchused": False, "kd_lr": 0.01, "weight_decay": 0.0001,
              "epochs": 200, "kd_T": 4},
    "coboosting": {"nz": 256, "ngf": 64, "g_steps": 30, "lr_g": 0.001, "synthesis_batch_size": 128,
                   "batch_size": 128, "adv": 1.0, "bn": 1.0, "oh": 1.0, "weighted": False,
                   "hs": 1.0, "wa_steps": 1, "mu": 0.01, "wdc": 0.99, "his": True,
                   "batchonly": False, "batchused": False, "kd_lr": 0.01, "weight_decay": 0.0001,
                   "epochs": 200, "kd_T": 4, "odseta": 8},
    "intactofl": {"gating_arch": "linear", "topk": 1.0, "nz": 256, "ngf": 64, "g_steps": 30,
                  "lr_g": 0.001, "synthesis_batch_size": 128, "batch_size": 128, "adv": 1.0,
                  "bn": 1.0, "oh": 1.0, "his": True, "batchonly": False, "batchused": False,
                  "epochs": 200, "gt_lr": 0.01, "weight_decay": 0.0001, "alpha": 0.01, "beta": 1,
                  "odseta": 8, "kd_T": 4},
}

setup_seed(config["seed"])

trainset, testset, client_idx_map = get_fl_dataset(
    config["dataset"]["data_name"], config["dataset"]["root_path"],
    config["client"]["num_clients"], config["distribution"]["type"],
    config["distribution"]["label_num_per_client"], config["distribution"]["alpha"])

test_loader = torch.utils.data.DataLoader(testset, batch_size=config["dataset"]["test_batch_size"], shuffle=True)

global_model = get_model(
    model_name=config["server"]["model_name"],
    num_classes=config["dataset"]["num_classes"],
    channels=config["dataset"]["channels"])

device = config["device"]

METHODS = {
    "FedAvg": OneShotFedAvg,
    "DENSE": OneShotDENSE,
    "CoBoosting": OneShotCoBoosting,
    "IntactOFL": OneShotIntactOFL,
    "Ensemble": OneShotEnsemble,
}

if METHOD not in METHODS:
    raise ValueError(f"Unknown method: {METHOD}. Available: {list(METHODS.keys())}")

total_ep = NUM_ROUNDS * LOCAL_EPOCHS
print(f"Running {METHOD} on {DATASET} alpha={ALPHA} GPU={GPU} budget={total_ep}ep ({NUM_ROUNDS}r x {LOCAL_EPOCHS}ep)")
METHODS[METHOD](trainset, test_loader, client_idx_map, config, global_model, device)
