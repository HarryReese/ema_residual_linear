#!/usr/bin/env bash

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-./dataset}
data_file=${DATA_FILE:-case6.csv}
pred_len=${PRED_LEN:-100}
seed=${SEED:-2026}
experiment_name=${EXPERIMENT_NAME:-core/case6_seq750_pred${pred_len}}

"$python_bin" -u run.py train \
  --seed "$seed" \
  --results_dir ./results \
  --experiment_name "$experiment_name" \
  --root_path "$data_root" \
  --data_path "$data_file" \
  --seq_len 750 \
  --pred_len "$pred_len" \
  --enc_in 3 \
  --d_model 96 \
  --d_ff 192 \
  --e_layers 1 \
  --dropout 0.001 \
  --seasonal_patch_len 64 \
  --seasonal_stride 32 \
  --ema_alpha 0.3 \
  --mamba_d_state 16 \
  --mamba_d_conv 4 \
  --mamba_expand 2 \
  --pm_channel_factor 2 \
  --pm_dropout 0.1 \
  --channel_beta_init 1.0 \
  --batch_size 48 \
  --learning_rate 0.00008 \
  --train_epochs 150 \
  "$@"
