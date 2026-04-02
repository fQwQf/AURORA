#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
/data1/tongjizhou/miniconda3/envs/fafi/bin/python test.py --cfp /tmp/cifar100_v18.yaml --algo OursV20
