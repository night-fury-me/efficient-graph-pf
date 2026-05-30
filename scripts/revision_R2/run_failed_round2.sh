#!/usr/bin/env bash
# Re-run only the 3 scripts that still failed after the first patch round:
#   R2_04 matfree (X_proj ctx)
#   R2_08 fullgraph (edge_vulnerability return signature)
#   R2_10 robust_arch (W_hidden recurrence)
# Cheapest first.
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
        echo "  FAIL in $(( $(date +%s) - t0 ))s -- see $log"
    fi
}

# R2_10 is the cheapest (50-node subgraphs, 2 archs, 2 datasets, 10 seeds)
run_one R2_10_robust_arch
# R2_04 next (synthetic + full datasets matrix-free; ~30 min)
run_one R2_04_matfree_error_bounds
# R2_08 last (full-graph matrix-free on Cora and Citeseer; ~60 min)
run_one R2_08_fullgraph_repro

echo
echo "=== Done ==="
ls -la results/revision_R2/*.csv
