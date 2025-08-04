#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=/Work21/2024/wangjunyu/SSL/fairseq/ \
python /Work21/2024/wangjunyu/SSL/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir ASDA/config \
    --config-name finetuning  \
    common.user_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA \
    common.seed=42 \
    checkpoint.save_dir=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_SPCv2 \
    checkpoint.restore_file=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/finetune_SPCv2/checkpoint_last.pt \
    dataset.batch_size=256 \
    criterion.log_keys=['correct'] \
    task.data=/CDShare3/2023/wangjunyu/audio/SPC_2 \
    task.spcv2_eval=True \
    task.target_length=128 \
    task.noise=true \
    task.roll_aug=true \
    optimization.lr=[0.0002] \
    optimizer.groups.default.lr_float=0.0002 \
    optimization.max_update=80000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=8000 \
    model.model_path=/Work21/2024/wangjunyu/SSL/fairseq/examples/ASDA/ASDA/model_ckpt/test/checkpoint_last.pt \
    model.num_classes=35 \
    model.spcv2_eval=true \
    model.mixup=0.8 \
    model.target_length=128 \
    model.mask_ratio=0.2 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN