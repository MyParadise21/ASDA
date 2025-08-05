#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python ASDA/feature_extract/feature_extract.py  \
    --source_file='ASDA/feature_extract/test.wav' \
    --target_file='ASDA/feature_extract/test.npy' \
    --model_dir='ASDA' \
    --checkpoint_dir='/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS2M/checkpoint_last.pt' \
    --granularity='frame' \
    --target_length=1024 \
    --mode='pretrain'