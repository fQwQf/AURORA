#!/bin/bash
set -e
cd /data1/tongjizhou/FAFI_ICML25
LOG_DIR=logs/neurips_alpha_sweep
mkdir -p $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)
for ALPHA in 0.1 0.3 0.5; do
  CFG=configs/neurips/CIFAR100_alpha${ALPHA}_ablation.yaml
  echo "[$(date)] Running AURORA alpha=${ALPHA}"
  CUDA_VISIBLE_DEVICES=0 python test.py --cfp $CFG --algo OursV14 \
    > ${LOG_DIR}/AURORA_alpha${ALPHA}_${TS}.log 2>&1
  echo "[$(date)] Done alpha=${ALPHA}"
done
echo "Alpha sweep complete. Logs in $LOG_DIR"
