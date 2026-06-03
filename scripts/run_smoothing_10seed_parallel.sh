#!/bin/bash
# 10-seed AEGIS-Conformal vs randomized-smoothing on Cora, 2-WAY parallel.
# MEASURED per-process peak ~9GB (dense torch.linalg.solve(I-J_z, J_A) + M=200
# reconverge buffers), NOT the ~2GB the script's help text guesses. So only 2 may
# run concurrently: 2x9=18GB < 23.5GB (5-way OOM'd at the S_c-build barrier).
# Each worker runs 5 seeds sequentially; a fresh python process per seed fully
# frees GPU memory between seeds, so peak stays at 2 concurrent S_c builds.
set -u
cd /home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fragmentation headroom
PY=.venv/bin/python

run_seeds() {
  for s in "$@"; do
    echo "[w$BASHPID] seed $s START $(date +%H:%M:%S)"
    if $PY scripts/exp_conformal_vs_smoothing.py \
         --dataset Cora --seed "$s" --device cuda \
         --M 200 --extrap-M 10000 --score aps \
         --out "cvs_cora_s${s}.csv" > "/tmp/cvs_s${s}.log" 2>&1; then
      echo "[w$BASHPID] seed $s OK $(date +%H:%M:%S)"
    else
      echo "[w$BASHPID] seed $s FAILED $(date +%H:%M:%S)"
    fi
  done
}

run_seeds 42 271 1729 3141 6561 &
sleep 5
run_seeds 137 314 2718 5772 9999 &
wait
echo "ALL_SEEDS_DONE $(date +%H:%M:%S)"
