export HF_HOME=/playpen/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME


CUDA_VISIBLE_DEVICES=4  python datasets/GeoCount/generate_sft_thinking.py --batch_size 16