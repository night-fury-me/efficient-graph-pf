#!/usr/bin/env python3
"""Re-run the 3 GAT-2 cells that OOM'd in fig_tau_heatmap, now that ExplicitGAT
uses sparse (edge-indexed) attention — verified == the dense reference to machine
precision by scripts/_verify_sparse_gat.py.

Appends GAT-2 x {Pubmed, WikiCS, Amazon Fraud} x 10 seeds to
results/tau_all_datasets.csv. Resumable (skips done dataset/arch/seed) and
preserves every existing row; writes atomically (temp + rename) after each cell.

Usage:  .venv/bin/python scripts/_rerun_gat_oom_cells.py
"""
from __future__ import annotations

import csv
import gc
import statistics as st
import sys
import time
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.exp_tau_all_datasets import (  # noqa: E402
    load_all_datasets, run_single, SEEDS,
)

CSV = PROJ / "results" / "tau_all_datasets.csv"
FIELDS = ["dataset", "architecture", "seed", "tau", "tau_weighted",
          "tau_weight_only", "p_at_5", "p_at_10", "p_at_20", "n_edges"]
TARGETS = ["Pubmed", "WikiCS", "Amazon Fraud"]
ARCH = "GAT-2"


def load_rows():
    return list(csv.DictReader(open(CSV))) if CSV.exists() else []


def save_rows(rows):
    tmp = CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    tmp.rename(CSV)  # atomic on POSIX


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    done = {(r["dataset"], r["architecture"], str(r["seed"])) for r in rows}
    print(f"device={device}  existing rows={len(rows)}", flush=True)
    print(f"targets: {ARCH} x {TARGETS} x {len(SEEDS)} seeds = {len(TARGETS)*len(SEEDS)} cells",
          flush=True)

    datasets = load_all_datasets()
    for ds in TARGETS:
        N = int(datasets[ds]["X"].shape[0])
        print(f"\n--- {ds} (N={N}) ---", flush=True)
        for seed in SEEDS:
            if (ds, ARCH, str(seed)) in done:
                print(f"  skip {ds} s{seed} (done)", flush=True)
                continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            r = run_single(ds, ARCH, datasets[ds], seed, device)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            if r is None:
                print(f"  {ds} s{seed}: returned None (still failed?)  "
                      f"{time.time()-t0:.0f}s peak={peak:.1f}GB", flush=True)
                continue
            rows.append({"dataset": ds, "architecture": ARCH, "seed": seed,
                         **{k: r[k] for k in FIELDS[3:]}})
            save_rows(rows)
            done.add((ds, ARCH, str(seed)))
            print(f"  {ds} s{seed}: tau={r['tau']:+.3f} tauW={r['tau_weighted']:+.3f} "
                  f"edges={r['n_edges']}  {time.time()-t0:.0f}s peak={peak:.1f}GB", flush=True)

    print("\n=== new GAT-2 cells (mean±sd over seeds) ===")
    for ds in TARGETS:
        ts = [float(r["tau"]) for r in rows
              if r["dataset"] == ds and r["architecture"] == ARCH]
        if ts:
            sd = st.stdev(ts) if len(ts) > 1 else 0.0
            print(f"  {ds:>13}: tau = {st.mean(ts):+.3f} ± {sd:.3f}  (n={len(ts)})")
    print(f"total rows now = {len(rows)}")


if __name__ == "__main__":
    main()
