#!/bin/bash
# Run extended baselines for AURORA paper comparison
# All use fafi conda env, all use existing datasets

set -e
cd /data1/tongjizhou/FAFI_ICML25
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate fafi

LOGDIR="logs/run_baselines"
mkdir -p "$LOGDIR"

INTACTDIR="/data1/tongjizhou/AURORA_bench/IntactOFL/source"
FUSEFLDIR="/data1/tongjizhou/AURORA_bench/FuseFL"
AFLDIR="/data1/tongjizhou/AURORA_bench/AFL"

CIFAR_DATA="/data1/tongjizhou/FAFI_ICML25/data/"
SVHN_DATA="$HOME/datasets/"
TINY_DATA="$HOME/datasets/"

run_intactofl_method() {
    local METHOD=$1
    local DATASET=$2
    local ALPHA=$3
    local GPU=$4
    local EXTRA="${5:-}"
    local LABEL="${METHOD}_${DATASET}_a${ALPHA}"
    local YAMLNAME="bench_${DATASET}_a${ALPHA}.yaml"

    # Generate config
    local DATANAME CLASSES IMGSIZE DATAROOT
    case $DATASET in
        cifar10) DATANAME="CIFAR10"; CLASSES=10; IMGSIZE=32; DATAROOT=$CIFAR_DATA ;;
        cifar100) DATANAME="CIFAR100"; CLASSES=100; IMGSIZE=32; DATAROOT=$CIFAR_DATA ;;
        svhn) DATANAME="SVHN"; CLASSES=10; IMGSIZE=32; DATAROOT=$SVHN_DATA ;;
        tiny) DATANAME="Tiny-ImageNet"; CLASSES=200; IMGSIZE=64; DATAROOT=$TINY_DATA ;;
    esac

    python3 -c "
import yaml
cfg = {
    'exp_name': '$LABEL',
    'dataset': {'data_name': '$DATANAME', 'root_path': '$DATAROOT',
                'train_batch_size': 128, 'test_batch_size': 256,
                'channels': 3, 'num_classes': $CLASSES, 'image_size': $IMGSIZE},
    'distribution': {'type': 'dirichlet', 'label_num_per_client': 2, 'alpha': $ALPHA},
    'client': {'num_clients': 5},
    'server': {'num_rounds': 100, 'frac_clients': 0.1, 'lr': 0.01,
               'local_epochs': 5, 'optimizer': 'sgd', 'momentum': 0.9,
               'weight_decay': 0.0001, 'loss_name': 'ce', 'model_name': 'resnet18',
               'aggregated_by_datasize': False, 'lr_decay_per_round': 0.998},
    'device': 'cuda:$GPU',
    'checkpoint': {'save_path': '/data1/tongjizhou/AURORA_bench/IntactOFL/checkpoints/$LABEL/', 'save_freq': 10},
    'resume': False, 'resume_best': False, 'seed': 42,
    'dense': {'nz': 256, 'ngf': 64, 'g_steps': 30, 'lr_g': 0.001, 'synthesis_batch_size': 128,
              'batch_size': 128, 'adv': 1.0, 'bn': 1.0, 'oh': 1.0, 'his': True,
              'batchonly': False, 'batchused': False, 'kd_lr': 0.01, 'weight_decay': 0.0001,
              'epochs': 200, 'kd_T': 4},
    'coboosting': {'nz': 256, 'ngf': 64, 'g_steps': 30, 'lr_g': 0.001, 'synthesis_batch_size': 128,
                   'batch_size': 128, 'adv': 1.0, 'bn': 1.0, 'oh': 1.0, 'weighted': False,
                   'hs': 1.0, 'wa_steps': 1, 'mu': 0.01, 'wdc': 0.99, 'his': True,
                   'batchonly': False, 'batchused': False, 'kd_lr': 0.01, 'weight_decay': 0.0001,
                   'epochs': 200, 'kd_T': 4, 'odseta': 8},
    'intactofl': {'gating_arch': 'linear', 'topk': 1.0, 'nz': 256, 'ngf': 64, 'g_steps': 30,
                  'lr_g': 0.001, 'synthesis_batch_size': 128, 'batch_size': 128, 'adv': 1.0,
                  'bn': 1.0, 'oh': 1.0, 'his': True, 'batchonly': False, 'batchused': False,
                  'epochs': 200, 'gt_lr': 0.01, 'weight_decay': 0.0001, 'alpha': 0.01, 'beta': 1,
                  'odseta': 8, 'kd_T': 4},
}
with open('$INTACTDIR/yaml_config/oneshot/$YAMLNAME', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Config written: $YAMLNAME')
"

    # Generate runner script that picks the right method
    local RUNFILE="/tmp/run_${LABEL}.py"
    cat > "$RUNFILE" << PYEOF
import sys
sys.path.insert(0, '$INTACTDIR')
from commona_libs import *
from dataset_helper import get_fl_dataset
from args import args_parser
from oneshot_algorithms import *

config = load_yaml_config('./yaml_config/oneshot/$YAMLNAME')
setup_seed(config['seed'])

trainset, testset, client_idx_map = get_fl_dataset(
    config['dataset']['data_name'], config['dataset']['root_path'],
    config['client']['num_clients'], config['distribution']['type'],
    config['distribution']['label_num_per_client'], config['distribution']['alpha'])

test_loader = torch.utils.data.DataLoader(testset, batch_size=config['dataset']['test_batch_size'], shuffle=True)

from models_lib.models import get_model
global_model = get_model(model_name=config['server']['model_name'],
                         num_classes=config['dataset']['num_classes'],
                         channels=config['dataset']['channels'])
device = config['device']

$METHOD(trainset, test_loader, client_idx_map, config, global_model, device)
PYEOF

    echo "[LAUNCH] $LABEL on GPU $GPU"
    nohup python "$RUNFILE" > "$LOGDIR/${LABEL}.log" 2>&1 &
    echo "  PID=$!"
}

# Phase 1: CIFAR-10 α=0.05 baselines on IntactOFL framework
run_intactofl_method OneShotFedAvg cifar10 0.05 1
run_intactofl_method OneShotDENSE cifar10 0.05 1
run_intactofl_method OneShotCoBoosting cifar10 0.05 1
run_intactofl_method OneShotIntactOFL cifar10 0.05 1

echo ""
echo "Baselines launched. Monitor: tail -f $LOGDIR/*.log"
