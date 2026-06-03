#!/bin/bash
# Single-stream (NO parallel) run of the 8 remaining smoothing seeds.
# Lesson learned: this workload is GPU-latency-bound (M=200 sequential fixed-point
# solves); 2 concurrent workers serialize on the kernel queue -> ~4x slower per seed,
# net ~2.7x slower than sequential. So run ONE at a time, uncontested (~12 min/seed).
# Each call writes its own --out CSV BEFORE the script's return, so a non-fatal
# self-check failure (single-seed conformal cov<0.9, a finite-sample artifact that
# averages out over 10 seeds) does NOT lose the CSV.
set -u
cd /home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for s in 271 314 1729 2718 3141 5772 6561 9999; do
  echo "[seq] seed $s START $(date +%H:%M:%S)"
  $PY scripts/exp_conformal_vs_smoothing.py \
      --dataset Cora --seed "$s" --device cuda \
      --M 200 --extrap-M 10000 --score aps \
      --out "cvs_cora_s${s}.csv" > "/tmp/cvs_s${s}.log" 2>&1
  rc=$?
  if [ -f "results/cvs_cora_s${s}.csv" ]; then
    echo "[seq] seed $s CSV OK (rc=$rc) $(date +%H:%M:%S)"
  else
    echo "[seq] seed $s NO CSV (rc=$rc) $(date +%H:%M:%S)"
  fi
done
echo "ALL_SEQ_DONE $(date +%H:%M:%S)"
