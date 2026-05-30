#!/bin/bash
# Run remaining R2 compute (G7-compute, G5, G6, G4) sequentially on the GPU.
# Each step writes its own log + CSV; the chain continues on partial failure.
set -u
cd "$(dirname "$0")/../.."
LOGD="results/revision_R2/logs"
mkdir -p "$LOGD"
PY=.venv/bin/python

stamp() { date "+%H:%M:%S"; }

run() {
  local name=$1; shift
  echo "[$(stamp)] === START $name ==="
  "$@" 2>&1 | tee "$LOGD/${name}.log" | tail -3
  echo "[$(stamp)] === END $name (rc=${PIPESTATUS[0]}) ==="
}

# 1. R2_04 re-run + postprocess (G7-compute)
run "R2_04_rerun_v2" $PY scripts/revision_R2/R2_04_matfree_error_bounds.py
run "R2_04_postprocess_v2" $PY scripts/revision_R2/postprocess_R2_04_csv.py

# 2. Amazon Photo full-graph (G5)
run "G5_amazon_fullgraph" $PY scripts/exp_amazon_fullgraph.py

# 3. Adaptive defense ablation (G6)
run "G6_adaptive_defense" $PY scripts/revision_R2/R2_13_adaptive_defense.py

# 4. PR-BCD baseline (G4)
run "G4_prbcd_baseline" $PY scripts/revision_R2/R2_11_prbcd_baseline.py

echo "[$(stamp)] === ALL DONE ==="
