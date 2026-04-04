#!/bin/bash

export HF_HOME=/playpen-shared/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export DEBUG_MODE="true"
export LOG_PATH="./logs/grpo_geocount_debug.log"

MODEL_PATH="Qwen/Qwen3-8B"
DATA_PATH="./datasets/GeoCount/shapes_2/train_task.json"
OUTPUT_DIR="./grpo_ckpt/geocount_qwen3"
RUN_NAME="qwen3-8b-lora-grpo-geocount"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $LOG_PATH)"

# Number of GPUs for training. When use_vllm=true, vLLM takes one extra GPU
# (the next one after the training GPUs), so set CUDA_VISIBLE_DEVICES to
# include one additional GPU beyond NUM_TRAIN_GPUS.
# Example: 4 training GPUs -> CUDA_VISIBLE_DEVICES=0,1,2,3,4 (4 is for vLLM)
NUM_TRAIN_GPUS=4
CUDA_VISIBLE_DEVICES=0,2,3,4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_TRAIN_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12346 \
    code/grpo_geocount.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --reward_funcs accuracy format \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    \
    --use_vllm true \
    --vllm_device "cuda:4" \
    --vllm_gpu_memory_utilization 0.7 \
    \
    --max_prompt_length 512 \
    --max_completion_length 8192 \
    --num_generations 8 \
    --temperature 1.0 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --num_train_epochs 1 \
    \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_dropout 0.05 \
    \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 3 \
    --report_to wandb \
    \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"