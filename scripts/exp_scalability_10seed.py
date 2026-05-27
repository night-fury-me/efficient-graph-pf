"""10-seed scalability experiment for AEGIS paper Table (dense vs matrix-free).

Tests N = 50, 200, 500, 1000, 2708 on Cora.
Reports timing (mean ± std) and peak GPU memory per configuration.

Usage:
    .venv/bin/python scripts/exp_scalability_10seed.py
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

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora
from iem.scalable import ScalableSensitivity

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
SUBGRAPH_SIZES = [50, 200, 500, 1000, 2708]
DENSE_LIMIT = 200
OUT_CSV = Path("results/exp_scalability_10seed.csv")


def train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed):
    torch.manual_seed(seed)
    model = IGNN(n_features, hidden=64, n_classes=n_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()
    return model


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def run_dense(model, Z_sub, ctx_sub, A_sub):
    def F_sub(z, c):
        return model.operator(z, c)

    J_z_flat, J_A, _ = _compute_structural_jacobian(F_sub, Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(F_sub, Z_sub, ctx_sub, J_z=J_z_flat, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] > 0:
        sigma_1 = float(torch.linalg.svdvals(S_c)[0])
    else:
        sigma_1 = 0.0
    return sigma_1


def run_matfree(model, Z_sub, ctx_sub):
    def F_sub(z, c):
        return model.operator(z, c)

    op = ScalableSensitivity(F_sub, Z_sub, ctx_sub)
    _, sigma, _ = op.top_k_svd(k=1, n_oversamples=5, n_power_iter=3)
    sigma_1 = float(sigma[0])
    return sigma_1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    _download_cora(Path("datasets/cora"))
    data = _load_cora(Path("datasets/cora"))
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    N_full = X.shape[0]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)

        model = train_ignn(
            X, A_hat, y, data["train_mask"],
            data["n_features"], data["n_classes"], device, seed,
        )
        with torch.no_grad():
            logits, Z_star, ctx = model(X, A_hat)
            pred = logits.argmax(dim=1)
            acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
        print(f"  acc={acc:.3f}", flush=True)

        for max_n in SUBGRAPH_SIZES:
            use_full = (max_n >= N_full)
            actual_n = N_full if use_full else max_n

            if use_full:
                A_sub = A_hat
                ctx_sub = ctx
                Z_sub = Z_star.detach()
            else:
                idx = extract_ego_subgraph(A_hat, max_nodes=max_n)
                actual_n = len(idx)
                A_sub = A_hat[idx][:, idx]
                ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
                with torch.no_grad():
                    Z_sub = Z_star[idx].clone()
                    for _ in range(200):
                        Z_new = model.operator(Z_sub, ctx_sub)
                        if (Z_new - Z_sub).norm() < 1e-7:
                            break
                        Z_sub = Z_new
                    Z_sub = Z_new

            n_edges = int((A_sub.abs() > 1e-10).sum() - actual_n) // 2
            row = {"seed": seed, "N": actual_n, "edges": n_edges}

            # --- Dense path ---
            if actual_n <= DENSE_LIMIT:
                reset_memory()
                try:
                    t0 = time.time()
                    sigma_dense = run_dense(model, Z_sub, ctx_sub, A_sub)
                    t_dense = time.time() - t0
                    mem_dense = peak_mem_mb()
                    row["dense_time"] = t_dense
                    row["dense_mem_mb"] = mem_dense
                    row["dense_sigma1"] = sigma_dense
                    row["dense_status"] = "OK"
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        row["dense_time"] = float("nan")
                        row["dense_mem_mb"] = float("nan")
                        row["dense_sigma1"] = float("nan")
                        row["dense_status"] = "OOM"
                        reset_memory()
                    else:
                        raise
            else:
                row["dense_time"] = float("nan")
                row["dense_mem_mb"] = float("nan")
                row["dense_sigma1"] = float("nan")
                row["dense_status"] = "OOM"

            # --- Matrix-free path ---
            reset_memory()
            try:
                t0 = time.time()
                sigma_mf = run_matfree(model, Z_sub, ctx_sub)
                t_mf = time.time() - t0
                mem_mf = peak_mem_mb()
                row["mf_time"] = t_mf
                row["mf_mem_mb"] = mem_mf
                row["mf_sigma1"] = sigma_mf
                row["mf_status"] = "OK"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    row["mf_time"] = float("nan")
                    row["mf_mem_mb"] = float("nan")
                    row["mf_sigma1"] = float("nan")
                    row["mf_status"] = "OOM"
                    reset_memory()
                else:
                    raise

            rows.append(row)

            d_str = f"{row['dense_time']:.1f}s" if row["dense_status"] == "OK" else "OOM"
            m_str = f"{row['mf_time']:.1f}s" if row["mf_status"] == "OK" else "OOM"
            print(f"  N={actual_n:>5} |E|={n_edges:>5}  dense={d_str:>8}  mf={m_str:>8}", flush=True)

            del A_sub, ctx_sub, Z_sub
            reset_memory()

        del model
        reset_memory()

    # --- Save CSV ---
    fields = [
        "seed", "N", "edges",
        "dense_time", "dense_mem_mb", "dense_sigma1", "dense_status",
        "mf_time", "mf_mem_mb", "mf_sigma1", "mf_status",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUT_CSV}", flush=True)

    # --- Summary table ---
    print(f"\n{'='*80}", flush=True)
    print("SCALABILITY SUMMARY (10 seeds)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'N':>5} | {'Dense (s)':>14} | {'MatFree (s)':>14} | {'Mem-Dense':>12} | {'Mem-MF':>12}", flush=True)
    print(f"  {'-'*67}", flush=True)

    for n_val in sorted(set(r["N"] for r in rows)):
        n_rows = [r for r in rows if r["N"] == n_val]

        d_times = [r["dense_time"] for r in n_rows if r["dense_status"] == "OK"]
        m_times = [r["mf_time"] for r in n_rows if r["mf_status"] == "OK"]
        d_mems = [r["dense_mem_mb"] for r in n_rows if r["dense_status"] == "OK"]
        m_mems = [r["mf_mem_mb"] for r in n_rows if r["mf_status"] == "OK"]

        if d_times:
            d_str = f"{np.mean(d_times):.1f}±{np.std(d_times):.1f}"
            dm_str = f"{np.mean(d_mems):.0f}MB"
        else:
            d_str = "OOM"
            dm_str = ">24GB"

        if m_times:
            m_str = f"{np.mean(m_times):.1f}±{np.std(m_times):.1f}"
            mm_str = f"{np.mean(m_mems):.0f}MB"
        else:
            m_str = "OOM"
            mm_str = ">24GB"

        print(f"  {n_val:>5} | {d_str:>14} | {m_str:>14} | {dm_str:>12} | {mm_str:>12}", flush=True)


if __name__ == "__main__":
    main()
