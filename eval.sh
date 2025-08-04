#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ASDA/evaluation/eval.py  \
    --label_file='ASDA/inference/labels.csv' \
    --eval_dir='/CDShare3/2023/wangjunyu/audio/AS2M' \
    --model_dir='ASDA' \
    --checkpoint_dir='/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS2M/checkpoint_last.pt' \
    --target_length=1024 \
    --device='cuda' \
    --batch_size=16
