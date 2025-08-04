#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ASDA/inference/inference.py  \
    --source_file='ASDA/inference/test.wav' \
    --label_file='ASDA/inference/labels.csv' \
    --model_dir='ASDA' \
    --checkpoint_dir='/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS2M/checkpoint_last.pt' \
    --target_length=1024 \
    --top_k_prediction=12 \
