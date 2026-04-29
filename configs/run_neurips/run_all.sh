#!/bin/bash
# Master run script for all missing NeurIPS experiments
# Usage: bash configs/run_neurips/run_all.sh
# Each experiment is launched in background on a specific GPU

set -e
cd /data1/tongjizhou/FAFI_ICML25

LOGDIR="logs/run_neurips"
mkdir -p "$LOGDIR"

activate_env() {
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate fafi
}

run_exp() {
    local GPU=$1
    local ALGO=$2
    local CONFIG=$3
    local LABEL=$4
    local LOGFILE="$LOGDIR/${LABEL}.log"
    
    # Check GPU memory
    local MEM_AVAIL=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "$MEM_AVAIL" -lt "8000" ]; then
        echo "[SKIP] $LABEL: GPU $GPU only has ${MEM_AVAIL}MB free (need 8000MB)"
        return 1
    fi
    
    echo "[LAUNCH] $LABEL on GPU $GPU (${MEM_AVAIL}MB free) -> $LOGFILE"
    
    # Modify device in config on-the-fly using python
    local TMP_CONFIG="/tmp/neurips_${LABEL}.yaml"
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['device'] = 'cuda:$GPU'
with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
" 2>/dev/null || cp "$CONFIG" "$TMP_CONFIG"
    
    nohup conda run --no-banner -n fafi python test.py --cfp "$TMP_CONFIG" --algo "$ALGO" > "$LOGFILE" 2>&1 &
    local PID=$!
    echo "  PID=$PID"
    echo "$PID" > "$LOGDIR/${LABEL}.pid"
}

activate_env

echo "=========================================="
echo "  NeurIPS Experiment Launch - $(date)"
echo "=========================================="

# GPU allocation plan:
# GPU 1 (~17GB free): Tiny-IN AURORA (100r, heaviest)
# GPU 3 (~18GB free): CIFAR-100 a=0.05 AURORA
# GPU 4 (~17GB free): CIFAR-100 a=0.1 AURORA
# After AURORA finishes, baselines can reuse GPUs

echo ""
echo "=== Phase 1: AURORA V24 runs ==="

run_exp 1 OursV24 configs/run_neurips/Tiny_a005_AURORA.yaml "Tiny_a005_AURORA" &
run_exp 3 OursV24 configs/run_neurips/CIFAR100_a005_AURORA.yaml "CIFAR100_a005_AURORA" &
run_exp 4 OursV24 configs/run_neurips/CIFAR100_a01_AURORA.yaml "CIFAR100_a01_AURORA" &

wait  # Wait for all background launches to complete

echo ""
echo "=== Phase 1 launched. Check progress: ==="
echo "  tail -f $LOGDIR/Tiny_a005_AURORA.log"
echo "  tail -f $LOGDIR/CIFAR100_a005_AURORA.log"
echo "  tail -f $LOGDIR/CIFAR100_a01_AURORA.log"
echo ""
echo "=== After Phase 1 completes, run Phase 2: ==="
echo "  bash configs/run_neurips/run_baselines.sh"
