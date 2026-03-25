#!/bin/bash
set -e

CFG="configs/neurips/ablation_base.yaml"
LOG_DIR="logs/neurips_ablation"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== NeurIPS Ablation Experiments ==="
echo "Config: $CFG"
echo "Log dir: $LOG_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

run_exp() {
    local algo=$1
    local extra_args=$2
    local logname="${LOG_DIR}/${algo}_${TIMESTAMP}.log"
    echo "[START] $algo $extra_args -> $logname"
    python test.py --cfp "$CFG" --algo "$algo" $extra_args > "$logname" 2>&1
    echo "[DONE]  $algo (exit=$?)"
}

echo "--- A6: No alignment (FAFI baseline) ---"
run_exp "OursV4" ""

echo "--- A2: Fixed lambda (ETF + fixed lambda, no uncertainty weighting) ---"
run_exp "OursV7" "--lambdaval 5.0"

echo "--- A3: AURORA w/o dynamic attenuation (V12 logic but no attenuation) ---"
run_exp "OursV12" ""

echo "--- AURORA canonical (V14) ---"
run_exp "OursV14" ""

echo "--- A5: Feature Collapse ablation ---"
run_exp "Ours_FeatureCollapse_Ablation" "--lambdaval 5.0"

echo "--- AURORA + FedAvg aggregation ---"
run_exp "AURORAFedAvg" ""

echo ""
echo "=== All ablation experiments complete ==="
echo "Logs in: $LOG_DIR"
