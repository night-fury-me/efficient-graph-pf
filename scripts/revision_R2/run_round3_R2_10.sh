#!/usr/bin/env bash
# Round-3 re-run of R2_10 robust_arch with spectral-normalized W (kappa <= 0.9).
# Waits for the current R2_04 / R2_08 background processes to finish so it does
# not contend for GPU memory.
set +e
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
LOGDIR="results/revision_R2/logs"
mkdir -p "$LOGDIR"

# Back up the round-2 robust_arch.csv (negative-tau result) for the record.
if [ -f results/revision_R2/robust_arch.csv ]; then
    cp -p results/revision_R2/robust_arch.csv \
          results/revision_R2/robust_arch_round2_unnormalized.csv
    echo "Backed up round-2 CSV to robust_arch_round2_unnormalized.csv"
fi

echo "Waiting for R2_04 / R2_08 to exit..."
while pgrep -f "R2_04_matfree\|R2_08_fullgraph" > /dev/null; do
    sleep 60
done
echo "GPU is free."

echo "=== R2_10 round-3 (spectral-norm W, kappa <= 0.9) ==="
t0=$(date +%s)
if $PY scripts/revision_R2/R2_10_robust_arch.py > "$LOGDIR/R2_10_robust_arch.log" 2>&1; then
    echo "  PASS in $(( $(date +%s) - t0 ))s"
else
    echo "  FAIL in $(( $(date +%s) - t0 ))s -- see $LOGDIR/R2_10_robust_arch.log"
fi

echo
echo "=== robust_arch.csv summary ==="
$PY <<'PY'
import pandas as pd
df = pd.read_csv("results/revision_R2/robust_arch.csv")
print(df.groupby(["dataset", "architecture"]).agg(
    n=("seed", "count"),
    mean_kappa=("kappa_Jz", "mean"),
    mean_tau=("tau_aegis_vs_brute", "mean"),
    std_tau=("tau_aegis_vs_brute", "std"),
).round(3))
PY
