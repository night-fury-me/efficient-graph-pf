#!/usr/bin/env bash
# Round-4 re-run of R2_10 with bug-fixed Kendall tau computation.
# Round 2 + 3 had a tau bug: aegis_rank was compared against gt_edges.index(e)
# (row-major iteration order), not the damage-sorted rank. Fixed in-place.
set +e
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
LOGDIR="results/revision_R2/logs"
mkdir -p "$LOGDIR"

# Preserve the round-3 (kappa<=0.9 but buggy tau) CSV for record-keeping
if [ -f results/revision_R2/robust_arch.csv ]; then
    cp -p results/revision_R2/robust_arch.csv \
          results/revision_R2/robust_arch_round3_buggy_tau.csv
    echo "Backed up round-3 CSV to robust_arch_round3_buggy_tau.csv"
fi

echo "Waiting for R2_04 / R2_08 to exit (GPU contention)..."
while pgrep -f "R2_04_matfree\|R2_08_fullgraph" > /dev/null; do
    sleep 60
done
echo "GPU is free."

echo "=== R2_10 round-4 (spectral-norm W, FIXED tau) ==="
t0=$(date +%s)
if $PY scripts/revision_R2/R2_10_robust_arch.py > "$LOGDIR/R2_10_robust_arch.log" 2>&1; then
    echo "  PASS in $(( $(date +%s) - t0 ))s"
else
    echo "  FAIL in $(( $(date +%s) - t0 ))s -- see $LOGDIR/R2_10_robust_arch.log"
fi

echo
echo "=== robust_arch.csv summary (round 4 - fixed) ==="
$PY <<'PY'
import pandas as pd
df = pd.read_csv("results/revision_R2/robust_arch.csv")
print(df.groupby(["dataset", "architecture"]).agg(
    n=("seed", "count"),
    mean_kappa=("kappa_Jz", "mean"),
    mean_tau=("tau_aegis_vs_brute", "mean"),
    std_tau=("tau_aegis_vs_brute", "std"),
    mean_acc=("test_accuracy", "mean"),
).round(3))
PY
