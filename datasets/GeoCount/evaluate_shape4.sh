CUDA_VISIBLE_DEVICES=4 python evaluate.py \
  --task_json ./shapes_4/task.json \
  --model_name_or_path /playpen-shared/haochenz/DeepSeek-R1-Distill-Qwen-7B \
  --output_path outputs/deepseek_r1_distill_qwen_7b_shapes3.jsonl \
  --use_chat_template \
  --dtype bfloat16 \
  --max_new_tokens 16384