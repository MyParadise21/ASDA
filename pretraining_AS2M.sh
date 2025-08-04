#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="/CDShare3/2023/wangjunyu/audio/pretrain"
echo "Using data path: $DATA_PATH"

PYTHONPATH=/Work21/2024/wangjunyu/SSL/fairseq/ \
python /Work21/2024/wangjunyu/SSL/fairseq/fairseq_cli/hydra_train.py \
    --config-dir ASDA/config \
    --config-name pretraining_AS2M \
    common.user_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA \
    checkpoint.save_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test \
    checkpoint.restore_file=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test/checkpoint_last.pt \
    distributed_training.distributed_world_size=4 \
    dataset.batch_size=12 \
    task.data=$DATA_PATH \
    task.h5_format=True