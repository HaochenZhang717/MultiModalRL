CUDA_VISIBLE_DEVICES=4 python evaluate_finetuned.py \
  --task_json ./shapes_2/test_task.json \
  --model_name_or_path /playpen-shared/haochenz/DeepSeek-R1-Distill-Qwen-7B \
  --adapter_path /playpen-shared/haochenz/MultiModalRL/grpo_ckpt/shapes_2/checkpoint-300 \
  --output_path outputs/finetuned_deepseek_r1_distill_qwen_7b_shapes2.jsonl \
  --use_chat_template \
  --dtype bfloat16 \
  --max_new_tokens 16384