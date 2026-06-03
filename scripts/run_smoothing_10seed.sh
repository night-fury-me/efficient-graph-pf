#!/bin/bash
# 10-seed AEGIS-Conformal vs randomized-smoothing head-to-head on Cora (dense n=200 case).
# Script overwrites results/conformal_vs_smoothing.csv each run, so we copy per-seed aside.
set -u
cd /home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN
SEEDS="42 137 271 314 1729 2718 3141 5772 6561 9999"
PY=.venv/bin/python
for s in $SEEDS; do
  echo "==================== seed $s ===================="
  $PY scripts/exp_conformal_vs_smoothing.py \
      --dataset Cora --seed "$s" --device auto \
      --M 200 --extrap-M 10000 --score aps || { echo "SEED $s FAILED"; continue; }
  cp results/conformal_vs_smoothing.csv "results/cvs_cora_s${s}.csv"
  echo "saved results/cvs_cora_s${s}.csv"
done
echo "ALL_SEEDS_DONE"
