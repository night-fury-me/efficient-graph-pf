#!/usr/bin/env bash
#
# Revision-R2 — run all 10 scripts in priority order.
#
# Total runtime estimate on a single RTX 4090:
#   R2_03 (stats reanalysis, no GPU)          : ~30 s
#   R2_07 (kappa direct, 5 datasets)          : ~25 min
#   R2_04 (matfree error bounds)              : ~40 min
#   R2_05 (PI baseline, case57+case118)       : ~15 min
#   R2_06 (LODF retarget + disagreement)      : ~30 min
#   R2_01 (GR-BCD baseline, 3 datasets)       : ~60 min
#   R2_02 (AGNNCert deterministic, 3 ds)      : ~50 min
#   R2_09 (iterative re-ranking, 3 ds)        : ~90 min (greedy is O(|E|^2))
#   R2_10 (robust-arch, 2 ds * 2 archs)       : ~45 min
#   R2_08 (full-graph repro, 2 ds)            : ~60 min
#   -------------------------------------------------
#   TOTAL                                     : ~7 hours
#
# Run individual scripts by skipping the rest.

# Continue past failures (each script logs its own output to results/revision_R2/)
set +e
cd "$(dirname "$0")/../.."

LOGDIR="results/revision_R2/logs"
mkdir -p "$LOGDIR"

run_one() {
    local name="$1"
    local log="$LOGDIR/${name}.log"
    echo "=== Launching $name (log: $log) ==="
    local t0=$(date +%s)
    if $PY "scripts/revision_R2/${name}.py" > "$log" 2>&1; then
        echo "  PASS in $(( $(date +%s) - t0 ))s"
    else
        echo "  FAIL in $(( $(date +%s) - t0 ))s — see $log"
    fi
}

PY=${PY:-.venv/bin/python}

echo "=== Revision-R2 — running 10 scripts in priority order ==="
echo "Project root: $(pwd)"
echo "Python: $PY"

# Cheap first
run_one R2_03_stats_reanalysis

# Priority 1
run_one R2_07_kappa_direct
run_one R2_04_matfree_error_bounds
run_one R2_05_pi_baseline
run_one R2_06_lodf_metric_retarget
run_one R2_01_grbcd_baseline
run_one R2_02_agnncert_comparison

# Priority 2
run_one R2_09_iterative_reranking
run_one R2_08_fullgraph_repro

# Priority 3
run_one R2_10_robust_arch

echo "=== All R2 scripts complete. Outputs in results/revision_R2/. ==="
echo
echo "Next step: integrate numbers into paper sections per the mapping in"
echo "scripts/revision_R2/README.md ('What still needs manual integration')."
