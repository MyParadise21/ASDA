#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

PYTHONPATH=/Work21/2024/wangjunyu/SSL/fairseq/ \
python /Work21/2024/wangjunyu/SSL/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir ASDA/config \
    --config-name finetuning  \
    common.user_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA \
    checkpoint.save_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS2M \
    checkpoint.restore_file=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS2M/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    distributed_training.distributed_world_size=1 \
    dataset.num_workers=4 \
    dataset.batch_size=48 \
    optimization.max_update=600000 \
    task.data=/CDShare3/2023/wangjunyu/audio/AS2M \
    task.h5_format=true \
    task.AS2M_finetune=true \
    task.weights_file=/CDShare3/2023/wangjunyu/audio/AS2M/weight_train_all.csv \
    task.target_length=1024 \
    task.roll_aug=true \
    model.model_path=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test/checkpoint_last.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN