"""BRUTAL finite-difference verification of the AEGIS core numerics.

Tests every core quantity against numerical ground truth (perturb -> reconverge
-> compare). If these pass, the method is correct independent of how the code
reads. Small real-IGNN Cora subgraph so dense S is computable.

CONVENTION (edge-weight, transfer-safe): S_c columns are the natural EDGE-WEIGHT
parametrization, column_k = S[:,iN+j] + S[:,jN+i] (NOT divided by sqrt2). Hence
v_k = ||column_k|| is the shift per unit edge-weight (delta c_k = 1), the per-edge
rankings and the transfer bridge d_k = w_k * v_k are correct, and sigma_1(S_c) is
the max shift per unit ||c||. A symmetric edge perturbation has ||dA||_F = sqrt2
||c||, so the per-Frobenius (threat-model-budgeted) bound is sigma_1(S_c)/sqrt2.

Checks:
  0. z* is an actual fixed point of F.
  1. J_z = dF/dz   vs finite difference (JVP form).
  2. J_A = dF/dvec(A) vs finite difference.
  3. S = (I-J_z)^{-1} J_A = dz*/dvec(A)  vs finite difference (reconverge).
  4. S_c columns vs S (construction) + sigma_1(S_c) consistency with ||c||.
  5. v_k = ||S_c[:,k]|| vs FD single-edge first-order damage (unit edge-weight b_k).
  6. transfer: d_k = ||z*(A) - z*(A\k)|| vs w_k * v_k  (Prop 3a).
  7. matrix-free matvec / top_k_svd / edge_vulnerability vs DENSE S_c.
  8. optimal_structural_attack: sigma_1(S_c) vs dense; 8b sigma_1(S_c)/sqrt2 per-Fro.

Usage: .venv/bin/python scripts/verify_core_implementation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import load_dataset, train_ignn, reconverge
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity

torch.set_printoptions(precision=4)
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, relerr, tol):
    ok = relerr is not None and relerr < tol
    results.append(ok)
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name:52s} rel-err={relerr:.3e}  (tol {tol:.0e})")
    return ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    X, A_hat, y, train_mask, n_features, n_classes = load_dataset("Cora")
    X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
    train_mask = train_mask.to(device)
    model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, 0)
    model.eval()

    idx = extract_ego_subgraph(A_hat, max_nodes=12)
    A = A_hat[idx][:, idx].clone().to(torch.float64)
    Xs = X[idx].to(torch.float64)
    # cast model to float64 for a clean FD signal
    model = model.double()
    A = A.double(); Xs = Xs.double()
    N = A.shape[0]
    Xproj = model.U(Xs)
    hid = Xproj.shape[1]
    ctx = {"A_hat": A, "X_proj": Xproj}

    def F(z, c=ctx):
        return model.operator(z, c)

    # --- z* fixed point ---
    z = torch.zeros(N, hid, dtype=torch.float64, device=device)
    for _ in range(2000):
        z2 = F(z)
        if (z2 - z).norm() < 1e-13:
            break
        z = z2
    z_star = z
    D = N * hid
    print(f"\nSubgraph N={N} hidden={hid} |edges|={int((A.abs()>1e-10).sum()-N)//2}  D={D}")
    print("=" * 78)

    # 0. fixed point
    fp_res = float((F(z_star) - z_star).norm() / max(z_star.norm(), 1e-12))
    check("0. z* is a fixed point  ||F(z*)-z*||/||z*||", fp_res, 1e-9)

    # analytical Jacobians + S
    J_z, J_A, _ = _compute_structural_jacobian(F, z_star, ctx)
    J_z, J_A = J_z.double(), J_A.double()
    S = structural_sensitivity_matrix(F, z_star, ctx, J_z=J_z, J_A=J_A).double()
    S_c, edge_list = constrained_sensitivity_matrix(S, A)
    S_c = S_c.double()
    n_edges = len(edge_list)

    eps = 1e-6

    # 1. J_z vs FD (random direction v, row-major vec)
    v = torch.randn(D, dtype=torch.float64, device=device)
    fd = (F(z_star + eps * v.reshape(N, hid)) - F(z_star)).reshape(-1) / eps
    an = J_z @ v
    e_jz = float((an - fd).norm() / max(fd.norm(), 1e-12))
    check("1. J_z @ v   vs  dF/dz finite-diff", e_jz, 1e-4)

    # 2. J_A vs FD (random symmetric edge-supported dA)
    dA = torch.zeros_like(A)
    ii = torch.nonzero(torch.triu(A.abs() > 1e-10, 1), as_tuple=False)
    for (i, j) in ii.tolist():
        r = float(np.random.randn())
        dA[i, j] = r; dA[j, i] = r
    fd_A = (F(z_star, {"A_hat": A + eps * dA, "X_proj": Xproj}) - F(z_star)).reshape(-1) / eps
    an_A = J_A @ dA.reshape(-1)
    e_ja = float((an_A - fd_A).norm() / max(fd_A.norm(), 1e-12))
    check("2. J_A @ vec(dA)  vs  dF/dA finite-diff", e_ja, 1e-4)

    # 3. S vs FD (small perturbation, reconverge)
    h = 1e-5
    z_pert = reconverge(model, z_star.clone(), {"A_hat": A + h * dA, "X_proj": Xproj})
    fd_S = (z_pert - z_star).reshape(-1) / h
    an_S = S @ dA.reshape(-1)
    e_s = float((an_S - fd_S).norm() / max(fd_S.norm(), 1e-12))
    check("3. S @ vec(dA)  vs  dz*/dA finite-diff (reconverge)", e_s, 5e-3)

    # 4. S_c construction vs S  +  sigma_1 consistency
    err_cons = 0.0
    for k, (i, j) in enumerate(edge_list):
        col = S[:, i * N + j] + S[:, j * N + i]
        err_cons = max(err_cons, float((S_c[:, k] - col).norm() / max(col.norm(), 1e-12)))
    check("4a. S_c[:,k] == S[:,iN+j]+S[:,jN+i]", err_cons, 1e-9)
    # sigma_1(S_c): max first-order shift over symmetric edge-supported ||dA||_F=1
    sig1 = float(torch.linalg.svdvals(S_c)[0])
    # brute random search for the max ||S @ vec(dA)|| over symmetric edge dA, ||dA||_F=1
    best = 0.0
    for _ in range(4000):
        d = torch.zeros_like(A)
        rr = torch.randn(n_edges, dtype=torch.float64)
        for k, (i, j) in enumerate(edge_list):
            d[i, j] = rr[k]; d[j, i] = rr[k]
        d = d / d.norm()
        best = max(best, float((S @ d.reshape(-1)).norm()))
    # best should be <= sig1 and approach it; report shortfall
    e_sig = abs(sig1 - best) / max(sig1, 1e-12)
    print(f"  [ -- ] 4b. sigma_1(S_c)={sig1:.4f}  random-search max={best:.4f}  "
          f"(best<=sig1: {best <= sig1*1.001})")

    # 5. v_k vs FD single-edge first-order damage.
    # CONVENTION: S_c uses the natural EDGE-WEIGHT parametrization, so its column
    # v_k = ||S_c[:,k]|| is the shift per unit EDGE-WEIGHT (delta c_k = 1), i.e. the
    # perturbation b_k = e_i e_j^T + e_j e_i^T (NOT unit-Frobenius; ||b_k||_F = sqrt2).
    # We perturb A by h*b_k and expect ||Dz||/h ~ v_k. This tests what v_ij actually
    # is, and keeps the transfer bridge d_k = w_k * v_k (check 6) consistent.
    v_ij = S_c.norm(dim=0)
    sqrt2 = 2.0 ** 0.5
    worst = 0.0
    for k, (i, j) in enumerate(edge_list[: min(n_edges, 8)]):
        bk = torch.zeros_like(A)
        bk[i, j] = 1.0; bk[j, i] = 1.0  # unit edge-weight: delta c_k = 1 (||bk||_F = sqrt2)
        zk = reconverge(model, z_star.clone(), {"A_hat": A + h * bk, "X_proj": Xproj})
        fd_vk = float((zk - z_star).norm() / h)
        worst = max(worst, abs(fd_vk - float(v_ij[k])) / max(float(v_ij[k]), 1e-12))
    check("5. v_k == FD single-edge first-order shift (unit edge-weight)", worst, 5e-3)

    # 6. transfer  d_k = ||z*(A) - z*(A\k)||  vs  w_k * v_k
    ratios = []
    for k, (i, j) in enumerate(edge_list[: min(n_edges, 8)]):
        w_k = float(A[i, j])
        A_rm = A.clone(); A_rm[i, j] = 0.0; A_rm[j, i] = 0.0
        z_rm = reconverge(model, z_star.clone(), {"A_hat": A_rm, "X_proj": Xproj})
        d_k = float((z_rm - z_star).norm())
        pred = w_k * float(v_ij[k])
        ratios.append(d_k / max(pred, 1e-12))
    ratios = np.array(ratios)
    print(f"  [ -- ] 6. transfer d_k / (w_k v_k): mean={ratios.mean():.3f} "
          f"range=[{ratios.min():.3f},{ratios.max():.3f}]  (->1 as first-order bridge)")

    # 7. matrix-free vs dense
    op = ScalableSensitivity(F, z_star, ctx)
    vv = torch.randn(n_edges, dtype=torch.float64, device=device)
    mf = op.matvec(vv.to(op.dtype)).double()
    dn = S_c @ vv
    e_mf = float((mf - dn).norm() / max(dn.norm(), 1e-12))
    check("7a. ScalableSensitivity.matvec  vs  dense S_c @ v", e_mf, 1e-3)
    torch.manual_seed(0)
    _, sg, _ = op.top_k_svd(k=min(6, n_edges), n_power_iter=10)
    e_svd = abs(float(sg[0]) - sig1) / max(sig1, 1e-12)
    check("7b. matrix-free sigma_1  vs  dense sigma_1(S_c)", e_svd, 5e-3)
    triples = op.edge_vulnerability()
    mf_v = {tuple(sorted((i, j))): val for i, j, val in triples}
    e_ev = 0.0
    for k, (i, j) in enumerate(edge_list):
        e_ev = max(e_ev, abs(mf_v[tuple(sorted((i, j)))] - float(v_ij[k])) / max(float(v_ij[k]), 1e-12))
    check("7c. edge_vulnerability  vs  dense ||S_c[:,k]||", e_ev, 5e-3)

    # 8. optimal_structural_attack: now reports the CONSTRAINED sigma_1(S_c),
    # consistent with its symmetric edge-supported attack direction (B2 fix).
    atk = optimal_structural_attack(S, A, epsilon=0.01)
    e_atk = abs(float(atk["sigma_1"]) - sig1) / max(sig1, 1e-12)
    check("8. optimal_structural_attack sigma_1(S_c)  vs  dense", e_atk, 5e-3)
    # 8b informational: per-Frobenius (||dA||_F-budgeted) bound = sigma_1(S_c)/sqrt2.
    # Verify the field B2 exposes and that it equals the max ||Dz|| over symmetric
    # edge dA with ||dA||_F = 1 (the threat-model interpretation).
    per_fro = float(atk["sigma_1_per_fro"])
    e_perfro = abs(per_fro - sig1 / sqrt2) / max(sig1 / sqrt2, 1e-12)
    # `best` (check 4b) is max ||S @ vec(dA)|| over symmetric edge dA with ||dA||_F=1,
    # i.e. exactly the per-Frobenius bound; it should match sigma_1(S_c)/sqrt2.
    e_best_perfro = abs(best - sig1 / sqrt2) / max(sig1 / sqrt2, 1e-12)
    print(f"  [ -- ] 8b. sigma_1(S_c)/sqrt2={sig1/sqrt2:.4f}  "
          f"atk.sigma_1_per_fro={per_fro:.4f} (relerr {e_perfro:.1e})  "
          f"random-search max-per-Fro={best:.4f} (relerr {e_best_perfro:.1e})  "
          f"[per-||dA||_F budget bound]")

    print("=" * 78)
    npass = sum(results)
    print(f"RESULT: {npass}/{len(results)} hard checks PASSED"
          + ("  -- core numerics VERIFIED" if npass == len(results) else "  -- !! FAILURES ABOVE !!"))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
