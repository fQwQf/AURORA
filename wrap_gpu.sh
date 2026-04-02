#!/bin/bash
export CUDA_VISIBLE_DEVICES=$GPU_ID
cd /data1/tongjizhou/FAFI_ICML25
/data1/tongjizhou/miniconda3/envs/fafi/bin/python test.py --cfp "$$CFP" --algo "$$ALgo" "$@
else
    echo "Usage: $0"
    echo "Available algorithms:"
    echo "  OursV14 OursV15 OursV16 OursV17 OursV18 OursV19 OursV20"
    echo "                 Ours_FeatureCollapse_Ablation"
    exit 1
fi

shift $GPU_ID $3
echo "GPU ${GPU_ID} (${GPU_ID:-?})"
echo "Available GPUs: $GPU_ID"
