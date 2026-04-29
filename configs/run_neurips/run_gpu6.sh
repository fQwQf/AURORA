#!/bin/bash
set -e
cd /data1/tongjizhou/FAFI_ICML25
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate fafi

LOGDIR="logs/run_baselines"
mkdir -p "$LOGDIR"

echo "=== FedProto Ens CIFAR10 500ep on GPU 6 ==="
python test.py --cfp /tmp/cifar10_a005_fedproto_ext.yaml --algo FedProto 2>&1 | tee "$LOGDIR/FedProto_CIFAR10_a005_ext500.log" || true

echo "=== FedETF Ens CIFAR10 500ep on GPU 6 ==="  
python test.py --cfp /tmp/cifar10_a005_fedetf_ext.yaml --algo FedETF 2>&1 | tee "$LOGDIR/FedETF_CIFAR10_a005_ext500.log" || true

echo "=== FedProto Ens SVHN 100ep on GPU 6 ==="
python test.py --cfp /tmp/svhn_a005_fedproto_ext.yaml --algo FedProto 2>&1 | tee "$LOGDIR/FedProto_SVHN_a005_ext100.log" || true

echo "=== FedETF Ens SVHN 100ep on GPU 6 ==="
python test.py --cfp /tmp/svhn_a005_fedetf_ext.yaml --algo FedETF 2>&1 | tee "$LOGDIR/FedETF_SVHN_a005_ext100.log" || true

echo "=== GPU 6 batch done ==="
