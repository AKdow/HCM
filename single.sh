export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=offline

python pretrain.py --config-name cfg_pretrain_math \
  2>&1 | tee ./pretrain_log/math_reg_$(date +%Y%m%d_%H%M%S).log   # 如果需要后台运行加上&
