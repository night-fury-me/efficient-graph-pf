#!/usr/bin/env bash
# Re-run only the 6 scripts that failed in the first pass.
# Fastest first so we get partial results early.
set +e
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
LOGDIR="results/revision_R2/logs"
mkdir -p "$LOGDIR"

run_one() {
    local name="$1"
    local log="$LOGDIR/${name}.log"
    echo "=== $name ==="
    local t0=$(date +%s)
    if $PY "scripts/revision_R2/${name}.py" > "$log" 2>&1; then
        echo "  PASS in $(( $(date +%s) - t0 ))s"
    else
        echo "  FAIL in $(( $(date +%s) - t0 ))s — see $log"
    fi
}

# Cheap pandapower scripts first (no GPU)
run_one R2_05_pi_baseline
run_one R2_06_lodf_metric_retarget

# GPU scripts (sequential — single GPU)
run_one R2_02_agnncert_comparison
run_one R2_04_matfree_error_bounds
run_one R2_10_robust_arch
run_one R2_08_fullgraph_repro

# R2_01 trailing-block bug fix (CSV already has complete data — quick re-run)
run_one R2_01_grbcd_baseline

echo
echo "=== Done ==="
ls -la results/revision_R2/*.csv
