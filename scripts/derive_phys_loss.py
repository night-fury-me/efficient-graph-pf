"""Derive per-epoch physics loss from history.csv + train.log.

The training loop logs only the *combined* loss when GNN_MSE_WEIGHT > 0:
    loss = phys + w * mse
Since `rmse² = mse` is also logged per epoch, we can recover phys exactly:
    phys = loss - w * rmse²

Usage:
    python scripts/derive_phys_loss.py <run_dir>

Example:
    python scripts/derive_phys_loss.py results/runs/260524-141231_a514
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def parse_w(train_log: Path) -> float:
    """GNN_MSE_WEIGHT is set in the launch script's `export` before training
    starts, so it doesn't appear in the train.log. Fall back to env var or
    a sensible default; allow caller to override via $GNN_MSE_WEIGHT."""
    w = os.environ.get("GNN_MSE_WEIGHT")
    if w is not None:
        return float(w)
    # Try to read from any GNN_MSE_WEIGHT mention in the log (defensive)
    text = train_log.read_text() if train_log.exists() else ""
    m = re.search(r"GNN_MSE_WEIGHT\s*[=:]\s*([\d.e+-]+)", text)
    if m:
        return float(m.group(1))
    # Default to 10 (our current LVN combined-loss recipe)
    print("WARNING: GNN_MSE_WEIGHT not in env or log; defaulting to 10.0", file=sys.stderr)
    return 10.0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/derive_phys_loss.py <run_dir>", file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    hist = run_dir / "artifacts" / "history.csv"
    log = run_dir / "train.log"

    if not hist.exists():
        print(f"history.csv missing: {hist}", file=sys.stderr)
        return 1

    w = parse_w(log)
    print(f"# Derived phys-loss using GNN_MSE_WEIGHT = {w}")
    print(f"# run: {run_dir.name}")
    print()
    print(f"{'epoch':>5} {'split':>5} {'combined_loss':>14} {'mse':>12} {'w*mse':>12} {'phys':>12} {'rmse':>10}")

    with hist.open() as f:
        header = f.readline().strip().split(",")
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            try:
                ep = parts[idx["epoch"]]
                split = parts[idx["split"]]
                loss = float(parts[idx["loss"]])
                rmse = float(parts[idx["rmse"]])
            except (ValueError, KeyError):
                continue  # skip NaN or malformed rows
            mse = rmse ** 2
            phys = loss - w * mse
            print(f"{ep:>5} {split:>5} {loss:>14.6e} {mse:>12.4e} {w*mse:>12.4e} {phys:>12.4e} {rmse:>10.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
