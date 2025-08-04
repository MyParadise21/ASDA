#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=/Work21/2024/wangjunyu/SSL/fairseq/ \
python /Work21/2024/wangjunyu/SSL/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir ASDA/config \
    --config-name finetuning  \
    common.user_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA \
    common.log_interval=100 \
    checkpoint.save_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/esc50-5/fold2 \
    checkpoint.restore_file=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/esc50-5/fold2/checkpoint_last.pt \
    dataset.batch_size=48 \
    criterion.log_keys=['correct'] \
    task.data=/CDShare3/2023/wangjunyu/audio/ESC_50/test05 \
    task.esc50_eval=True \
    task.target_length=512 \
    task.roll_aug=true \
    optimization.max_update=4000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=400 \
    model.model_path=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test/checkpoint_last.pt \
    model.num_classes=50 \
    model.esc50_eval=true \
    model.mixup=0.0 \
    model.target_length=512 \
    model.mask_ratio=0.4 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN