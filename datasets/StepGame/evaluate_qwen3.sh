#export HF_HOME=/playpen-shared/haochenz/hf_cache
#export TRANSFORMERS_CACHE=$HF_HOME
#export HF_DATASETS_CACHE=$HF_HOME



python eval_qwen3.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data_nhop6/test.jsonl \
    --batch-size 8 \
    --limit 100 \
    --output-dir "results/qwen2.5-7B-instruct/nhop6"