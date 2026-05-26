#!/bin/bash
# Run all remaining experiments sequentially (they share the GPU)
# Usage: ./scripts/run_all_experiments.sh 2>&1 | tee results/experiment_log.txt

set -e
cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python

echo "=========================================="
echo "B2: Attack baselines [P1.2]"
echo "=========================================="
$PYTHON scripts/exp_attack_baselines.py

echo ""
echo "=========================================="
echo "B6: Breach rates at all epsilon [P2.2]"
echo "=========================================="
$PYTHON scripts/exp_breach_rates.py

echo ""
echo "=========================================="
echo "B3: Scalability on larger graphs [P1.4]"
echo "=========================================="
$PYTHON scripts/exp_scalability_large.py

echo ""
echo "=========================================="
echo "B1: Cross-dataset/architecture tau [P1.3]"
echo "=========================================="
$PYTHON scripts/exp_tau_all_datasets.py

echo ""
echo "=========================================="
echo "B7: Power grid enhancements [P2.5]"
echo "=========================================="
$PYTHON scripts/exp_power_grid_enhanced.py

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Results in results/"
ls -la results/*.csv results/*.json 2>/dev/null
