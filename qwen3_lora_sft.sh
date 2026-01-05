#!/bin/bash

set -x
nvidia-smi && sudo fuser -kv /dev/nvidia*

MODEL_PATH=Qwen/Qwen3-32B

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target all \
    --dataset tb_plus_12_19_claude,tb_plus_12_19_gemini \
    --template qwen3 \
    --cutoff_len 20000 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir $HOME/research_nfs/jasonqi_weights/llama_factory/qwen3-32b-tb-plus-claude-gemini-12-29/lora/sft \
    --logging_steps 1 \
    --save_steps 20 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.5e-5 \
    --num_train_epochs 15.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json
