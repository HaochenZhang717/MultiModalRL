export CUDA_VISIBLE_DEVICES=0,2,3,4
torchrun \
  --nproc_per_node=4 \
  code/train_grpo.py