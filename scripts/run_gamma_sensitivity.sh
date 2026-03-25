#!/bin/bash
set -e
cd /data1/tongjizhou/FAFI_ICML25
LOG_DIR=logs/neurips_gamma_sensitivity
mkdir -p $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)
for GAMMA in 0.000001 0.00001 0.0001 0.001; do
  CFG=configs/neurips/ablation_base.yaml
  echo "[$(date)] Running AURORA gamma=${GAMMA}"
  CUDA_VISIBLE_DEVICES=0 python test.py --cfp $CFG --algo OursV14 \
    --gamma_reg ${GAMMA} \
    > ${LOG_DIR}/AURORA_gamma${GAMMA}_${TS}.log 2>&1
  echo "[$(date)] Done gamma=${GAMMA}"
done
echo "Gamma sensitivity complete. Logs in $LOG_DIR"
