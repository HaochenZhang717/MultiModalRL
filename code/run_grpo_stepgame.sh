#!/bin/bash

export HF_HOME=/playpen-shared/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export DEBUG_MODE="true"
export LOG_PATH="./logs/grpo_stepgame_debug.log"

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="../datasets/StepGame/data_nhop6/train.jsonl"
OUTPUT_DIR="./grpo_ckpt/stepgame_qwen2.5"
RUN_NAME="qwen2.5-7b-lora-grpo-stepgame-nhop6"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $LOG_PATH)"

# Number of GPUs for training. When use_vllm=true, vLLM takes one extra GPU
# (the next one after the training GPUs), so set CUDA_VISIBLE_DEVICES to
# include one additional GPU beyond NUM_TRAIN_GPUS.
# Example: 4 training GPUs -> CUDA_VISIBLE_DEVICES=0,1,2,3,4 (4 is for vLLM)
NUM_TRAIN_GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_TRAIN_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12347 \
    grpo_stepgame.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --story_key story \
    --reward_funcs accuracy \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 1.0 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant" \
    --max_steps 500 \
    \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_dropout 0.05 \
    \
    --bf16 true \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters false \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 3 \
    --report_to wandb \
    \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
