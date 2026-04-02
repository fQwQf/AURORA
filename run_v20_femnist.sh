#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
/data1/tongjizhou/miniconda3/envs/fafi/bin/python test.py --cfp /tmp/femnist_aurora_v18.yaml --algo OursV20
