#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=/Work21/2024/wangjunyu/SSL/fairseq/ \
python /Work21/2024/wangjunyu/SSL/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir ASDA/config \
    --config-name finetuning  \
    common.user_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA \
    checkpoint.save_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS20K \
    checkpoint.restore_file=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_test_AS20K/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=32 \
    task.data=/CDShare3/2023/wangjunyu/audio/AS20K \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=60000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=6000 \
    model.model_path=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test/checkpoint_last.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN