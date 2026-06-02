#!/bin/bash
# Per-job wrapper run by the cluster scheduler on each host.
# Isolated working dir (so per-dataset shards don't collide) with datasets/
# and configs/ symlinked in, so the experiment scripts' RELATIVE paths
# (e.g. Path("datasets/cora")) resolve while outputs stay per-job.
# Records the TRUE python exit code in done.flag.
# usage: run_job.sh <label> <script.py> "<args>"
BASE=/proj/ciptmp/up89uvox/aegis
VENV=/proj/ciptmp/up89uvox/my_project_venv
label="$1"; script="$2"; args="$3"; seeds="$4"
D="$BASE/results/cluster/$label"
mkdir -p "$D/results" "$D/paper/review"
ln -sfn "$BASE/datasets" "$D/datasets"
ln -sfn "$BASE/configs"  "$D/configs"
cd "$D" || { echo "DONE 98" > "$D/done.flag"; exit 98; }
rm -f done.flag
[ -n "$seeds" ] && export AEGIS_SEEDS="$seeds"
# shellcheck disable=SC2086  (args intentionally word-split)
"$VENV/bin/python" "$BASE/scripts/$script" $args > run.log 2>&1
echo "DONE $?" > done.flag
