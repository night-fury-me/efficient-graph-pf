"""B3/P1.4 — Scalability on larger graphs + subgraph vs full-graph ranking.

Part A: Matrix-free timing on Amazon Photo (N=7,650) and Pubmed (N=19,717).
Part B: Subgraph vs full-graph ranking comparison on Cora (Kendall tau, P@k).

Seeds (timing):  [42, 137, 271]
Seeds (ranking): [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python scripts/exp_scalability_large.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_amazon import _load_amazon
from iem.examples.ignn_citeseer_pubmed import _load_planetoid
from iem.examples.ignn_cora import IGNN, _load_cora
from iem.scalable import ScalableSensitivity

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
TIMING_SEEDS = SEEDS[:3]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_ignn(data, device, seed, epochs=200):
    """Train IGNN with early stopping, return model + fixed-point context."""
    set_seed(seed)
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A_hat)
                val_acc = float(
                    (logits.argmax(1)[data["val_mask"]] == y[data["val_mask"]])
                    .float().mean()
                )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    return model, Z_star, ctx


# ------------------------------------------------------------------
# Part A: Large-graph matrix-free timing
# ------------------------------------------------------------------

def run_part_a(device):
    """Time the matrix-free pipeline on Amazon Photo and Pubmed (full graph)."""
    print("\n" + "=" * 80)
    print("PART A: Large-graph matrix-free timing")
    print("=" * 80)

    datasets = [
        ("Amazon Photo", lambda: _load_amazon(Path("datasets/amazon_photo"))),
        ("Pubmed", lambda: _load_planetoid("pubmed", Path("datasets/pubmed"))),
    ]

    results_a = []

    for ds_name, loader in datasets:
        print(f"\n--- {ds_name} ---", flush=True)
        data = loader()
        N = data["N"]
        print(f"  N={N}, features={data['n_features']}, classes={data['n_classes']}", flush=True)

        seed_times = []
        seed_mems = []

        for seed in TIMING_SEEDS:
            print(f"  Seed {seed} ...", end=" ", flush=True)

            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                model, Z_star, ctx = train_ignn(data, device, seed)

                # Matrix-free S_c on FULL GRAPH
                t0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                op = ScalableSensitivity(
                    model.operator, Z_star, ctx,
                    neumann_terms=0, neumann_tol=1e-6,
                )
                U, sigma, Vh = op.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                peak_mem = (
                    torch.cuda.max_memory_allocated() / 1e6
                    if torch.cuda.is_available() else 0.0
                )

                sigma_1 = float(sigma[0])
                seed_times.append(elapsed)
                seed_mems.append(peak_mem)
                print(f"time={elapsed:.1f}s  mem={peak_mem:.0f}MB  sigma_1={sigma_1:.4f}", flush=True)

                results_a.append({
                    "dataset": ds_name,
                    "N": N,
                    "seed": seed,
                    "time_s": elapsed,
                    "peak_mem_MB": peak_mem,
                    "sigma_1": sigma_1,
                    "status": "OK",
                })

                # Cleanup
                del model, Z_star, ctx, op, U, sigma, Vh
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_msg = str(e)[:80]
                print(f"OOM at N={N}: {err_msg}", flush=True)
                results_a.append({
                    "dataset": ds_name,
                    "N": N,
                    "seed": seed,
                    "time_s": float("nan"),
                    "peak_mem_MB": float("nan"),
                    "sigma_1": float("nan"),
                    "status": f"OOM: {err_msg}",
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if seed_times:
            mean_t = np.mean(seed_times)
            std_t = np.std(seed_times)
            mean_m = np.mean(seed_mems)
            print(f"  => {ds_name}: {mean_t:.1f} +/- {std_t:.1f}s, peak_mem={mean_m:.0f}MB")

    return results_a


# ------------------------------------------------------------------
# Part B: Subgraph vs full-graph ranking comparison on Cora
# ------------------------------------------------------------------

def run_part_b(device):
    """Compare edge vulnerability rankings: full-graph (matrix-free) vs subgraph (dense)."""
    print("\n" + "=" * 80)
    print("PART B: Subgraph vs full-graph ranking on Cora")
    print("=" * 80)

    data = _load_cora(Path("datasets/cora"))
    print(f"  Cora: N={data['N']}", flush=True)

    results_b = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ...", flush=True)
        set_seed(seed)

        model, Z_star, ctx = train_ignn(data, device, seed)
        A_hat = data["A_hat"].to(device)

        # --- Full-graph matrix-free vulnerability ---
        print("    Full-graph matrix-free ...", end=" ", flush=True)
        t0 = time.perf_counter()
        op_full = ScalableSensitivity(
            model.operator, Z_star, ctx,
            neumann_terms=0, neumann_tol=1e-6,
        )
        full_vulns = op_full.edge_vulnerability()
        t_full = time.perf_counter() - t0
        print(f"{t_full:.1f}s, {len(full_vulns)} edges", flush=True)

        # Build full-graph vulnerability dict: (min_i, max_j) -> score
        full_vuln_dict = {}
        for i, j, v in full_vulns:
            key = (min(i, j), max(i, j))
            full_vuln_dict[key] = v

        # --- Subgraph (50-node BFS) dense vulnerability ---
        print("    Subgraph dense ...", end=" ", flush=True)
        idx = extract_ego_subgraph(A_hat, max_nodes=50)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
        Z_sub = Z_star[idx].clone()

        # Reconverge to subgraph fixed point
        with torch.no_grad():
            for _ in range(200):
                Z_new = model.operator(Z_sub, ctx_sub)
                if (Z_new - Z_sub).norm() < 1e-7:
                    break
                Z_sub = Z_new
        Z_sub = Z_new

        t0 = time.perf_counter()
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        t_sub = time.perf_counter() - t0
        print(f"{t_sub:.1f}s, {len(edge_list)} edges", flush=True)

        if not edge_list:
            print("    No edges in subgraph, skipping", flush=True)
            continue

        # Subgraph per-edge vulnerability
        sub_vuln = {}
        for k, (i, j) in enumerate(edge_list):
            sub_vuln[(i, j)] = float(S_c[:, k].norm())

        # --- Map subgraph edges to global node IDs ---
        idx_list = idx.tolist()
        # Build mapping: subgraph_local_edge -> global_edge
        sub_scores = []
        full_scores = []
        for (si, sj), sv in sub_vuln.items():
            gi, gj = idx_list[si], idx_list[sj]
            gkey = (min(gi, gj), max(gi, gj))
            if gkey in full_vuln_dict:
                sub_scores.append(sv)
                full_scores.append(full_vuln_dict[gkey])

        n_common = len(sub_scores)
        if n_common < 3:
            print(f"    Only {n_common} common edges, skipping", flush=True)
            continue

        # Kendall tau
        tau, p_val = kendalltau(sub_scores, full_scores)

        # P@5 and P@10
        sub_ranked = sorted(range(n_common), key=lambda i: sub_scores[i], reverse=True)
        full_ranked = sorted(range(n_common), key=lambda i: full_scores[i], reverse=True)

        def precision_at_k(ranked_a, ranked_b, k):
            k = min(k, len(ranked_a))
            top_a = set(ranked_a[:k])
            top_b = set(ranked_b[:k])
            return len(top_a & top_b) / k

        p_at_5 = precision_at_k(sub_ranked, full_ranked, 5)
        p_at_10 = precision_at_k(sub_ranked, full_ranked, 10)

        print(f"    Common edges: {n_common}, tau={tau:+.3f}, P@5={p_at_5:.2f}, P@10={p_at_10:.2f}", flush=True)

        results_b.append({
            "seed": seed,
            "n_common_edges": n_common,
            "kendall_tau": tau,
            "p_at_5": p_at_5,
            "p_at_10": p_at_10,
            "time_full_s": t_full,
            "time_sub_s": t_sub,
        })

        # Cleanup
        del model, Z_star, ctx, op_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_b


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Part A
    results_a = run_part_a(device)

    # Part B
    results_b = run_part_b(device)

    # --- Save CSVs ---
    csv_a = results_dir / "exp_scalability_large_timing.csv"
    with open(csv_a, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "N", "seed", "time_s", "peak_mem_MB", "sigma_1", "status",
        ])
        writer.writeheader()
        writer.writerows(results_a)
    print(f"\nPart A results saved to {csv_a}")

    csv_b = results_dir / "exp_scalability_large_ranking.csv"
    with open(csv_b, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "n_common_edges", "kendall_tau", "p_at_5", "p_at_10",
            "time_full_s", "time_sub_s",
        ])
        writer.writeheader()
        writer.writerows(results_b)
    print(f"Part B results saved to {csv_b}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n--- Part A: Large-graph timing (mean +/- std over 3 seeds) ---")
    for ds_name in ["Amazon Photo", "Pubmed"]:
        rows = [r for r in results_a if r["dataset"] == ds_name and r["status"] == "OK"]
        if rows:
            times = [r["time_s"] for r in rows]
            mems = [r["peak_mem_MB"] for r in rows]
            Ns = rows[0]["N"]
            print(f"  {ds_name:<15} N={Ns:>6}  time={np.mean(times):.1f}+/-{np.std(times):.1f}s  "
                  f"peak_mem={np.mean(mems):.0f}MB")
        else:
            oom_rows = [r for r in results_a if r["dataset"] == ds_name]
            if oom_rows:
                print(f"  {ds_name:<15} N={oom_rows[0]['N']:>6}  OOM")

    print("\n--- Part B: Subgraph vs full-graph ranking (Cora, 10 seeds) ---")
    if results_b:
        taus = [r["kendall_tau"] for r in results_b if not np.isnan(r["kendall_tau"])]
        p5s = [r["p_at_5"] for r in results_b]
        p10s = [r["p_at_10"] for r in results_b]
        print(f"  Kendall tau: {np.mean(taus):.3f} +/- {np.std(taus):.3f}")
        print(f"  P@5:         {np.mean(p5s):.3f} +/- {np.std(p5s):.3f}")
        print(f"  P@10:        {np.mean(p10s):.3f} +/- {np.std(p10s):.3f}")
        t_fulls = [r["time_full_s"] for r in results_b]
        t_subs = [r["time_sub_s"] for r in results_b]
        print(f"  Full-graph time: {np.mean(t_fulls):.1f} +/- {np.std(t_fulls):.1f}s")
        print(f"  Subgraph time:   {np.mean(t_subs):.1f} +/- {np.std(t_subs):.1f}s")
    else:
        print("  No results collected.")


if __name__ == "__main__":
    sys.exit(main() or 0)
