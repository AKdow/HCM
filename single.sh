export CUDA_VISIBLE_DEVICES=7
export WANDB_MODE=offline

python pretrain.py \
  data_path=data/MATH-401/exp6 \
  epochs=1 \
  eval_interval=1 \
  global_batch_size=8 \
  lr=5e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=0.0 \
  puzzle_emb_weight_decay=0.0