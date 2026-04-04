export CUDA_VISIBLE_DEVICES=0,2,3,4
export HF_HOME=/playpen-shared/haochenz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

torchrun \
  --nproc_per_node=4 \
  code/train_grpo.py