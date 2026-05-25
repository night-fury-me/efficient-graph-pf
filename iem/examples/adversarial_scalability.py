"""Scalability analysis: IEM adversarial analysis at varying subgraph sizes.

Measures wall-clock time and tightness for N = 20, 50, 100, 200 node subgraphs
on Cora. Reports Jacobian, sensitivity matrix, SVD, and total time.

Usage:
    .venv/bin/python -m iem.examples.adversarial_scalability
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    certified_shift_bound,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Loading Cora + Training IGNN ===", flush=True)
    _download_cora(Path("datasets/cora"))
    data = _load_cora(Path("datasets/cora"))
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    import torch.nn.functional as F_func
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(100):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    print(f"  test_acc={acc:.3f}\n")

    # --- Scalability scan ---
    subgraph_sizes = [20, 50, 100, 200]
    results = []

    for max_n in subgraph_sizes:
        print(f"--- N={max_n} ---", flush=True)

        idx = extract_ego_subgraph(A_hat, max_nodes=max_n)
        S_size = len(idx)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

        # Reconverge
        Z_sub = Z_star[idx].clone()
        with torch.no_grad():
            for _ in range(200):
                Z_new = model.operator(Z_sub, ctx_sub)
                if (Z_new - Z_sub).norm() < 1e-7:
                    break
                Z_sub = Z_new
        Z_sub = Z_new

        n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
        D = Z_sub.numel()
        print(f"  nodes={S_size}, edges={n_edges}, D={D}", flush=True)

        def F_sub(z, c):
            return model.operator(z, c)

        # Time: state Jacobian J_z
        t0 = time.time()
        from iem.ift import compute_jacobian
        def F_z(z):
            return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
        J_z = compute_jacobian(F_z, Z_sub)
        t_jz = time.time() - t0

        # Time: structural Jacobian J_A
        t0 = time.time()
        _, J_A, _ = _compute_structural_jacobian(F_sub, Z_sub, ctx_sub)
        t_ja = time.time() - t0

        # Time: sensitivity matrix S = (I-J_z)^{-1} J_A
        t0 = time.time()
        S = structural_sensitivity_matrix(F_sub, Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
        t_solve = time.time() - t0

        # Time: constrained SVD
        t0 = time.time()
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        if S_c.shape[1] > 0:
            sigma_c = torch.linalg.svdvals(S_c)
            sigma_1_c = float(sigma_c[0])
        else:
            sigma_1_c = 0.0
        t_svd = time.time() - t0

        t_total = t_jz + t_ja + t_solve + t_svd

        # Rho
        rho = spectral_radius(F_z, Z_sub)

        # Tightness (quick, single epsilon)
        tight_results = validate_bound_tightness(
            F_sub, model, Z_sub, ctx_sub, S, epsilons=[0.01], n_random=3,
        )
        constr_tight = tight_results[0]["constr_tightness"] if tight_results else 0

        print(f"  rho={rho:.4f}, sigma_1_c={sigma_1_c:.2f}, tightness={constr_tight:.3f}")
        print(f"  J_z: {t_jz:.1f}s | J_A: {t_ja:.1f}s | solve: {t_solve:.1f}s | SVD: {t_svd:.2f}s | total: {t_total:.1f}s")

        results.append({
            "N": S_size, "edges": n_edges, "D": D,
            "t_jz": t_jz, "t_ja": t_ja, "t_solve": t_solve, "t_svd": t_svd,
            "t_total": t_total, "rho": rho, "sigma_1_c": sigma_1_c,
            "constr_tight": constr_tight,
        })

        # Memory cleanup for large subgraphs
        del J_z, J_A, S, S_c
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Summary table ---
    print(f"\n{'='*80}", flush=True)
    print("SCALABILITY ANALYSIS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'N':>5} {'|E|':>5} {'D':>6} {'J_z':>7} {'J_A':>7} {'solve':>7} {'SVD':>6} {'total':>7} {'tight':>6} {'rho':>6}", flush=True)
    print(f"  {'-'*72}", flush=True)
    for r in results:
        print(f"  {r['N']:>5} {r['edges']:>5} {r['D']:>6} "
              f"{r['t_jz']:>6.1f}s {r['t_ja']:>6.1f}s {r['t_solve']:>6.2f}s {r['t_svd']:>5.2f}s "
              f"{r['t_total']:>6.1f}s {r['constr_tight']:>6.3f} {r['rho']:>6.3f}", flush=True)

    # Scaling analysis
    if len(results) >= 2:
        r0, r1 = results[0], results[-1]
        n_ratio = r1["D"] / r0["D"]
        t_ratio = r1["t_total"] / r0["t_total"] if r0["t_total"] > 0.01 else float("inf")
        print(f"\n  Scaling: D grew {n_ratio:.1f}x, time grew {t_ratio:.1f}x")
        print(f"  Expected O(D^3) scaling: {n_ratio**3:.0f}x")
        print(f"  Practical limit: ~200 nodes (D~12800) in ~{results[-1]['t_total']:.0f}s" if r1['N'] >= 200 else "")


if __name__ == "__main__":
    sys.exit(main() or 0)
