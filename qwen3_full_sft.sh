#!/bin/bash

set -x
nvidia-smi && sudo fuser -kv /dev/nvidia*

MODEL_PATH=Qwen/Qwen3-8B

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset tb_plus_12_19_claude,tb_plus_12_19_gemini \
    --template qwen3 \
    --cutoff_len 24000 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir $HOME/research_nfs/jasonqi_weights/llama_factory/qwen3-8b-tb-plus-claude-gemini-1-7/full/sft \
    --logging_steps 1 \
    --save_steps 40 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model true \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 20.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json