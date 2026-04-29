#!/bin/bash
# Extended Table 2 experiments batch runner
set -e
cd /data1/tongjizhou/FAFI_ICML25
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate fafi

LOGDIR="logs/run_baselines"
mkdir -p "$LOGDIR"

run_ext() {
    local METHOD=$1 DATASET=$2 ALPHA=$3 GPU=$4 ROUNDS=$5 LEP=$6
    local LABEL="${METHOD}_${DATASET}_a${ALPHA}_ext"
    local TOTAL=$((ROUNDS * LEP))
    echo "[$(date)] Starting $LABEL (${ROUNDS}r x ${LEP}ep = ${TOTAL}ep) on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python configs/run_neurips/baseline_runner_extended.py \
        "$METHOD" "$DATASET" "$ALPHA" 0 "$ROUNDS" "$LEP" \
        > "$LOGDIR/${LABEL}.log" 2>&1
    echo "[$(date)] Finished $LABEL"
}

# GPU 1: SVHN 100ep experiments
run_ext FedAvg SVHN 0.05 1 100 1
run_ext DENSE SVHN 0.05 1 100 1
run_ext CoBoosting SVHN 0.05 1 100 1
run_ext IntactOFL SVHN 0.05 1 100 1

echo "=== GPU 1 SVHN batch done ==="

# GPU 4: CIFAR10 500ep experiments
run_ext FedAvg CIFAR10 0.05 4 100 5
run_ext DENSE CIFAR10 0.05 4 100 5
run_ext CoBoosting CIFAR10 0.05 4 100 5
run_ext IntactOFL CIFAR10 0.05 4 100 5

echo "=== All extended experiments done ==="
