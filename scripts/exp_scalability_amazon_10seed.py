"""10-seed scalability for Amazon Photo (N=7650) — SVD-based pipeline.

At N=7650 with |E|=119K, the per-edge column computation is impractical.
The scalable pipeline uses randomized SVD to extract top-k singular
triplets, from which vulnerability rankings follow via |Vh[0,:]|.
This matches the methodology behind the paper's 363s timing.

Usage:
    .venv/bin/python scripts/exp_scalability_amazon_10seed.py
"""
from __future__ import annotations

import csv
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F_func
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.examples.ignn_cora import IGNN
from iem.examples.ignn_amazon import _download_amazon, _load_amazon
from iem.scalable import ScalableSensitivity

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
OUT_CSV = Path("results/exp_scalability_amazon_10seed.csv")


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    _download_amazon(Path("datasets/amazon_photo"))
    data = _load_amazon(Path("datasets/amazon_photo"))
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    N = X.shape[0]

    rows = []

    for seed in SEEDS:
        print(f"\n--- Amazon Photo seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(200):
            model.train()
            logits, _, _ = model(X, A_hat)
            loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            logits, Z_star, ctx = model(X, A_hat)
            acc = float((logits.argmax(1)[data["test_mask"]] == y[data["test_mask"]]).float().mean())
        print(f"  acc={acc:.3f}", flush=True)

        def F_op(z, c):
            return model.operator(z, c)

        reset_memory()
        try:
            t0 = time.time()
            op = ScalableSensitivity(F_op, Z_star.detach(), ctx)
            _, sigma, Vh = op.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)
            sigma_1 = float(sigma[0])
            t_total = time.time() - t0
            mem_mb = peak_mem_mb()
            n_edges = op.num_edges
            status = "OK"
            print(f"  N={N} |E|={n_edges}  mf={t_total:.1f}s  σ₁={sigma_1:.1f}  mem={mem_mb:.0f}MB", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                t_total = float("nan")
                mem_mb = float("nan")
                sigma_1 = float("nan")
                n_edges = 0
                status = "OOM"
                print(f"  N={N}  OOM", flush=True)
                reset_memory()
            else:
                raise

        rows.append({
            "dataset": "Amazon Photo", "seed": seed, "N": N, "edges": n_edges,
            "mf_time": t_total, "mf_mem_mb": mem_mb, "mf_sigma1": sigma_1, "mf_status": status,
        })

        del model, Z_star, ctx
        reset_memory()

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "N", "edges", "mf_time", "mf_mem_mb", "mf_sigma1", "mf_status"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {OUT_CSV}", flush=True)

    # Summary
    ok = [r for r in rows if r["mf_status"] == "OK"]
    if ok:
        times = [r["mf_time"] for r in ok]
        mems = [r["mf_mem_mb"] for r in ok]
        print(f"\nAmazon Photo N={N} (10 seeds):")
        print(f"  MatFree: {np.mean(times):.1f}±{np.std(times):.1f}s")
        print(f"  Memory:  {np.mean(mems):.0f}MB")


if __name__ == "__main__":
    main()
