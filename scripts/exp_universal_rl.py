#!/usr/bin/env python
"""AEGIS-Universal: operator-agnostic validation on an RL VALUE fixed point.

Thesis
------
AEGIS's constrained-sensitivity machinery

    S = (I - J_z)^{-1} J_A   (resolvent form)

plus its three diagnostics (per-edge ranking, SVD-optimal direction,
sensitivity magnitude) need ONLY a contractive fixed-point operator F and its
Jacobian pair (J_z, J_A). They do NOT need the bespoke ReLU-IGNN.

We prove this by running the SAME code path used for the IGNN
(``iem.adversarial.structural_sensitivity_matrix``) on the Bellman policy-
evaluation operator, whose fixed point is the RL value function.

Structural identity
-------------------
Policy evaluation  V^pi = r^pi + gamma P^pi V^pi  is a contraction (modulus
gamma < 1) EXACTLY like the IGNN's z* = F(z*, A).  The operator is

    F(V, ctx={P, r}) = r + gamma * (P @ V),

so  J_z = dF/dV = gamma P  (spectral radius gamma, guaranteed contractive)
and J_P = dF/dvec(P) is the structural Jacobian w.r.t. the transition graph.
By the IFT,

    S_value = dV/dvec(P) = (I - gamma P)^{-1} J_P,

the SAME resolvent form as AEGIS, with (I - gamma P)^{-1} playing the role of
(I - J_z)^{-1}.  Per-edge scores v_k = ||S_value[:, k]|| answer "which
transition edge most shifts the value function" (structural credit assignment);
the leading right singular vector of S_value is the most value-disruptive
transition perturbation.

Code path
---------
PRIMARY (no IGNN, no code change): we feed F directly to the EXISTING
``structural_sensitivity_matrix(F, z_star, ctx, A_key='P')``.  Its helper
``_compute_structural_jacobian`` perturbs every entry of ctx['P'] by finite
difference -> J_P, builds J_z by autograd row-backward, and solves
(I - J_z)^{-1} J_P.  This is byte-for-byte the IGNN path.

DOCUMENTED DEVIATION (per-edge constrained columns only):
``constrained_sensitivity_matrix`` and ``ScalableSensitivity._edges_to_delta_A``
assume SYMMETRIC edge perturbations (delta A[i,j] = delta A[j,i], upper triangle
only) -- correct for an undirected GNN graph, WRONG for a directed row-
stochastic transition graph.  The minimal change is exactly: replace the
symmetric column  S[:, i*N+j] + S[:, j*N+i]  with the directed column
S[:, i*N+j], iterating over nonzeros of P (both triangles, no i<j filter).
We implement that directed edge basis here (``directed_edge_columns``); every
other quantity comes from the unmodified library S.

Verifications (all finite-difference, gold standard)
----------------------------------------------------
A. S_value vs FD: random row-consistent structural delta dP on existing
   transitions; check  S_value @ vec(dP)  ~  (V' - V)  to <1e-4 relative.
B. Transfer bridge (analog of Prop:transfer, eq:transfer  d_k = w_k v_k + R_k):
   per directed edge k=(i,j),  d_k = ||V(P) - V(P without edge k)||  vs the
   edge-weighted score  w_k * v_k  (w_k=P[i,j], v_k=||S_value[:,k]||).
   Edge removal uses FIXED-NORMALIZATION masking -- set P[i,j] -> 0 with the
   single-entry delta dP[i,j] = -w_k and NO row renormalization -- exactly the
   paper's modelled deletion ([delta A]_ij = -w_k, degree matrix held fixed;
   the paper notes recompute-normalization adds only an O(d_i^-1) rescaling).
   Report ratio d_k/(w_k v_k) (-> 1) and Kendall tau of the edge-weighted
   w_k*v_k ranking vs brute-force d_k (the paper's headline score), plus the
   unweighted v_k ranking tau. A recompute-normalization variant is reported as
   a documented secondary.
C. SVD-optimal direction: leading right singular vector of S_value (directed
   edge basis) yields larger ||Delta V|| than equal-norm random structural
   perturbations.

Cheap: CPU, S=60, no training, no GPU.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# EXISTING AEGIS library -- the same functions the IGNN uses.
from iem.adversarial import (
    structural_sensitivity_matrix,
    _compute_structural_jacobian,
    optimal_structural_attack,
)
from iem.certify import spectral_radius

torch.set_default_dtype(torch.float64)  # tight FD comparisons


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------
def build_mdp(S: int, succ: int, gamma: float, seed: int):
    """Random row-stochastic transition graph P^pi, reward r^pi.

    Returns P (S,S, sparse ~succ/row, row sums = 1), r (S,), gamma.
    """
    rng = np.random.default_rng(seed)
    P = torch.zeros(S, S)
    for i in range(S):
        js = rng.choice(S, size=succ, replace=False)
        w = torch.from_numpy(rng.random(succ))
        w = w / w.sum()
        for k, j in enumerate(js):
            P[i, int(j)] = w[k]
    # numerical row-normalisation guard
    P = P / P.sum(dim=1, keepdim=True)
    r = torch.from_numpy(rng.random(S))
    return P, r, gamma


def value_function(P: torch.Tensor, r: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact V = (I - gamma P)^{-1} r."""
    I = torch.eye(P.shape[0], dtype=P.dtype)
    return torch.linalg.solve(I - gamma * P, r)


def make_operator(gamma: float):
    """Bellman operator F(z, ctx) = r + gamma P z, in AEGIS's (z, ctx) form."""
    def F(z: torch.Tensor, ctx: dict) -> torch.Tensor:
        z = z.reshape(-1)
        return ctx["r"] + gamma * (ctx["P"] @ z)
    return F


# ---------------------------------------------------------------------------
# Directed-edge basis  (the documented minimal deviation)
# ---------------------------------------------------------------------------
def directed_edge_columns(S_full: torch.Tensor, P: torch.Tensor, tol: float = 1e-10):
    """Directed analog of constrained_sensitivity_matrix.

    For each existing transition (i,j) with P[i,j] > tol, take the single
    column S_full[:, i*N+j] (row-major vec). Returns S_c (D, |E|) and the
    ordered edge list [(i,j), ...].  This is the asymmetric counterpart of the
    library's  S[:, i*N+j] + S[:, j*N+i]  (which is only correct for symmetric
    undirected perturbations).
    """
    N = P.shape[0]
    cols, edges = [], []
    for i in range(N):
        for j in range(N):
            if P[i, j].abs() > tol:
                cols.append(S_full[:, i * N + j])
                edges.append((i, j))
    S_c = torch.stack(cols, dim=1) if cols else torch.zeros(S_full.shape[0], 0)
    return S_c, edges


# ---------------------------------------------------------------------------
# Verification A: S_value vs finite difference
# ---------------------------------------------------------------------------
def verify_fd_jvp(F, V, ctx, P, r, gamma, S_full, n_trials, seed):
    """S_value @ vec(dP) ~ (V' - V) for row-consistent structural deltas.

    Each dP is supported ONLY on existing transitions and has ZERO row sums, so
    P' = P + dP stays row-stochastic (the value-function analog of the IGNN's
    fixed-normalisation edge perturbation). vec is row-major to match S_full.
    """
    rng = np.random.default_rng(seed)
    N = P.shape[0]
    mask = (P.abs() > 1e-10)
    errs = []
    for _ in range(n_trials):
        dP = torch.zeros(N, N)
        # random delta on existing edges, then project each row to zero sum
        noise = torch.from_numpy(rng.standard_normal((N, N))) * mask
        for i in range(N):
            row_mask = mask[i]
            m = int(row_mask.sum())
            if m == 0:
                continue
            row = noise[i].clone()
            row[row_mask] = row[row_mask] - row[row_mask].mean()  # zero-sum on support
            dP[i] = row * row_mask
        scale = 1e-5 / (dP.norm() + 1e-30)
        dP = dP * scale  # tiny structural step

        # library prediction
        pred = (S_full @ dP.reshape(-1))
        # ground truth
        Pp = P + dP
        Vp = value_function(Pp, r, gamma)
        truth = Vp - V
        rel = float((pred - truth).norm() / (truth.norm() + 1e-30))
        errs.append(rel)
    return float(np.median(errs)), float(np.max(errs)), errs


# ---------------------------------------------------------------------------
# Verification B: per-edge transfer bridge
# ---------------------------------------------------------------------------
def remove_edge_fixed_norm(P: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """P with directed edge (i,j) removed by FIXED-NORMALIZATION masking.

    Single-entry delta dP[i,j] = -w_k, no row renormalization. This is the
    paper's modelled deletion in prop:transfer(a): [delta A]_ij = -w_k with the
    degree matrix held fixed, so ||Delta z*|| ~ w_k v_k by the S_c construction.
    """
    Pp = P.clone()
    Pp[i, j] = 0.0
    return Pp


def remove_edge_renormalized(P: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Recompute-normalization variant (documented secondary).

    Drop edge (i,j) then renormalize row i to sum 1 -- the paper's noted
    O(d_i^-1) incident-edge rescaling. Reported only for completeness; it does
    NOT match the single-entry v_k and so is expected to disagree.
    """
    Pp = P.clone()
    Pp[i, j] = 0.0
    s = Pp[i].sum()
    if s > 1e-12:
        Pp[i] = Pp[i] / s
    return Pp


def verify_transfer_bridge(P, r, gamma, V, S_full, edges):
    """For each directed edge k=(i,j): d_k = ||V - V(P\\k)|| vs w_k * v_k.

    PRIMARY bridge uses fixed-normalization removal (paper's prop:transfer).
    Also computes the recompute-normalization variant for completeness.
    """
    from scipy.stats import kendalltau

    N = P.shape[0]
    rows = []
    for (i, j) in edges:
        w_k = float(P[i, j])
        v_k = float(S_full[:, i * N + j].norm())
        # primary: fixed-normalization (single-entry) removal
        Vp = value_function(remove_edge_fixed_norm(P, i, j), r, gamma)
        d_k = float((V - Vp).norm())
        # secondary: recompute-normalization removal
        Vp2 = value_function(remove_edge_renormalized(P, i, j), r, gamma)
        d_k_renorm = float((V - Vp2).norm())
        ratio = d_k / (w_k * v_k + 1e-30)
        rows.append({"i": i, "j": j, "w_k": w_k, "v_k": v_k,
                     "wk_vk": w_k * v_k, "d_k": d_k,
                     "d_k_renorm": d_k_renorm, "ratio": ratio})
    ratios = np.array([x["ratio"] for x in rows])
    vk = np.array([x["v_k"] for x in rows])
    wkvk = np.array([x["wk_vk"] for x in rows])
    dk = np.array([x["d_k"] for x in rows])
    dk_renorm = np.array([x["d_k_renorm"] for x in rows])
    tau_vk, _ = kendalltau(vk, dk)              # unweighted score ranking
    tau_wkvk, _ = kendalltau(wkvk, dk)          # edge-weighted (paper headline)
    tau_renorm, _ = kendalltau(wkvk, dk_renorm)  # secondary variant
    summary = {
        "n_edges": len(rows),
        "ratio_median": float(np.median(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios)),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "kendall_tau_vk_dk": float(tau_vk),
        "kendall_tau_wkvk_dk": float(tau_wkvk),
        "kendall_tau_renorm_secondary": float(tau_renorm),
    }
    return summary, rows


# ---------------------------------------------------------------------------
# Verification C: SVD-optimal direction vs random
# ---------------------------------------------------------------------------
def verify_svd_direction(S_c, edges, P, r, gamma, V, n_random, eps, seed):
    """Leading right singular vector of S_c gives larger ||Delta V|| than
    equal-norm random directed-structural perturbations.

    A unit edge-vector u (|E|,) maps to a directed dP on existing transitions;
    we then RE-PROJECT each row to zero sum so P' stays row-stochastic and the
    nonlinear ground-truth ||V(P')-V|| is a fair structural comparison.
    """
    rng = np.random.default_rng(seed)
    N = P.shape[0]
    E = len(edges)
    eidx = torch.tensor(edges, dtype=torch.long)  # (E,2)

    def edges_to_dP(u: torch.Tensor) -> torch.Tensor:
        dP = torch.zeros(N, N)
        dP[eidx[:, 0], eidx[:, 1]] = u
        return dP

    def row_zero_sum(dP: torch.Tensor) -> torch.Tensor:
        mask = (P.abs() > 1e-10)
        out = torch.zeros_like(dP)
        for i in range(N):
            rm = mask[i]
            if int(rm.sum()) == 0:
                continue
            row = dP[i].clone()
            row[rm] = row[rm] - row[rm].mean()
            out[i] = row * rm
        return out

    def deltaV_linear(u):
        return float((S_c @ u).norm())

    def deltaV_true(u_unit):
        dP = edges_to_dP(u_unit) * eps
        dP = row_zero_sum(dP)
        # keep the requested norm after projection
        n = dP.norm()
        if n > 1e-30:
            dP = dP * (eps / n)
        Pp = P + dP
        Vp = value_function(Pp, r, gamma)
        return float((Vp - V).norm())

    # SVD-optimal direction on the directed S_c
    U, sigma, Vh = torch.linalg.svd(S_c, full_matrices=False)
    v1 = Vh[0]
    v1 = v1 / v1.norm()
    svd_lin = deltaV_linear(v1)
    svd_true = deltaV_true(v1)

    rand_lin, rand_true = [], []
    for _ in range(n_random):
        u = torch.from_numpy(rng.standard_normal(E))
        u = u / u.norm()
        rand_lin.append(deltaV_linear(u))
        rand_true.append(deltaV_true(u))
    rand_lin = np.array(rand_lin)
    rand_true = np.array(rand_true)

    return {
        "sigma_1": float(sigma[0]),
        "svd_linear_dV": svd_lin,
        "svd_true_dV": svd_true,
        "rand_linear_dV_max": float(rand_lin.max()),
        "rand_linear_dV_mean": float(rand_lin.mean()),
        "rand_true_dV_max": float(rand_true.max()),
        "rand_true_dV_mean": float(rand_true.mean()),
        "margin_linear_vs_randmax": svd_lin / (rand_lin.max() + 1e-30),
        "margin_true_vs_randmax": svd_true / (rand_true.max() + 1e-30),
        "margin_true_vs_randmean": svd_true / (rand_true.mean() + 1e-30),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    S = int(os.environ.get("RL_S", 60))
    succ = int(os.environ.get("RL_SUCC", 6))
    gamma = float(os.environ.get("RL_GAMMA", 0.9))
    seed = int(os.environ.get("RL_SEED", 0))

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 72)
    print(f"AEGIS-Universal RL value fixed point | S={S} succ/state={succ} "
          f"gamma={gamma} seed={seed}")
    print("=" * 72)

    P, r, gamma = build_mdp(S, succ, gamma, seed)
    V = value_function(P, r, gamma)
    F = make_operator(gamma)
    ctx = {"P": P, "r": r}

    # ---- contractivity / fixed-point sanity (critique-inspect) ----
    fp_res = float((F(V, ctx) - V).norm())
    rho = spectral_radius(lambda z: F(z.reshape(V.shape), ctx).reshape(-1), V)
    row_sum_err = float((P.sum(dim=1) - 1.0).abs().max())
    print(f"[sanity] fixed-point residual ||F(V)-V|| = {fp_res:.3e}")
    print(f"[sanity] rho(J_z) = {rho:.6f}  (== gamma => contraction, resolvent well-defined)")
    print(f"[sanity] max row-sum deviation of P = {row_sum_err:.3e} (row-stochastic)")

    # ---- EXISTING AEGIS code path: S_value = (I - J_z)^{-1} J_P ----
    Jz, JA, _ = _compute_structural_jacobian(F, V, ctx, A_key="P")
    jz_err = float((Jz - gamma * P).abs().max())
    S_full = structural_sensitivity_matrix(F, V, ctx, A_key="P", J_z=Jz, J_A=JA)
    # closed-form cross-check: S_value should equal (I - gamma P)^{-1} J_P
    I = torch.eye(S, dtype=P.dtype)
    S_closed = torch.linalg.solve(I - gamma * P, JA)
    s_closed_err = float((S_full - S_closed).norm() / (S_closed.norm() + 1e-30))
    print(f"[path ] EXISTING structural_sensitivity_matrix used (A_key='P'), "
          f"S_value shape {tuple(S_full.shape)}")
    print(f"[path ] max|J_z - gamma*P| = {jz_err:.3e}  (FD Jacobian == analytic)")
    print(f"[path ] ||S_lib - (I-gamma P)^-1 J_P|| / ||.|| = {s_closed_err:.3e}")

    # directed edge basis (documented minimal deviation)
    S_c, edges = directed_edge_columns(S_full, P)
    print(f"[path ] directed edge basis: |E| = {len(edges)} columns")

    # ---- Verification A: S_value vs FD ----
    medA, maxA, _ = verify_fd_jvp(F, V, ctx, P, r, gamma, S_full,
                                  n_trials=20, seed=seed + 1)
    print(f"[A] S_value vs FD JVP   median rel err = {medA:.3e}  "
          f"max = {maxA:.3e}  (target < 1e-4)")

    # ---- Verification B: transfer bridge (fixed-normalization, paper's prop) ----
    bridge, bridge_rows = verify_transfer_bridge(P, r, gamma, V, S_full, edges)
    print(f"[B] transfer bridge (fixed-norm)  ratio d_k/(w_k v_k): "
          f"median {bridge['ratio_median']:.4f} "
          f"mean {bridge['ratio_mean']:.4f} "
          f"[p10 {bridge['ratio_p10']:.3f}, p90 {bridge['ratio_p90']:.3f}]")
    print(f"[B] Kendall tau (edge-weighted w_k v_k vs d_k) = "
          f"{bridge['kendall_tau_wkvk_dk']:.4f}  <- paper headline score")
    print(f"[B] Kendall tau (unweighted v_k vs d_k)       = "
          f"{bridge['kendall_tau_vk_dk']:.4f}   "
          f"| recompute-norm secondary tau = {bridge['kendall_tau_renorm_secondary']:.4f}")

    # ---- Verification C: SVD vs random ----
    svd = verify_svd_direction(S_c, edges, P, r, gamma, V,
                               n_random=200, eps=1e-4, seed=seed + 2)
    print(f"[C] SVD-optimal dV (linear) = {svd['svd_linear_dV']:.3e}  vs "
          f"random max {svd['rand_linear_dV_max']:.3e}  "
          f"(margin x{svd['margin_linear_vs_randmax']:.2f})")
    print(f"[C] SVD-optimal dV (true)   = {svd['svd_true_dV']:.3e}  vs "
          f"random max {svd['rand_true_dV_max']:.3e}  "
          f"(margin x{svd['margin_true_vs_randmax']:.2f}, "
          f"vs mean x{svd['margin_true_vs_randmean']:.2f})")

    # ---- verdict ----
    ok_fd = maxA < 1e-4
    ok_bridge_ratio = abs(bridge["ratio_median"] - 1.0) < 0.2  # ratio -> 1
    ok_bridge_tau = bridge["kendall_tau_wkvk_dk"] > 0.8        # edge-weighted, paper headline
    ok_svd = svd["margin_true_vs_randmax"] > 1.0
    verdict = ok_fd and ok_bridge_ratio and ok_bridge_tau and ok_svd
    print("-" * 72)
    print(f"VERDICT operator-agnostic on Bellman value fixed point: "
          f"{'YES' if verdict else 'NO'}")
    print(f"  FD<1e-4: {ok_fd} ({maxA:.1e}) | ratio~1: {ok_bridge_ratio} "
          f"(median {bridge['ratio_median']:.3f}) | "
          f"tau(w_k v_k)>0.8: {ok_bridge_tau} ({bridge['kendall_tau_wkvk_dk']:.3f}) | "
          f"SVD>rand: {ok_svd} (x{svd['margin_true_vs_randmax']:.2f})")
    print("=" * 72)

    # ---- write CSV ----
    out_csv = ROOT / "scripts" / "results_universal_rl.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["S_states", S])
        w.writerow(["succ_per_state", succ])
        w.writerow(["gamma", gamma])
        w.writerow(["seed", seed])
        w.writerow(["n_edges", len(edges)])
        w.writerow(["fixed_point_residual", fp_res])
        w.writerow(["rho_Jz", rho])
        w.writerow(["row_sum_err", row_sum_err])
        w.writerow(["jz_minus_gammaP_maxabs", jz_err])
        w.writerow(["S_lib_vs_closedform_relerr", s_closed_err])
        w.writerow(["A_fd_jvp_median_relerr", medA])
        w.writerow(["A_fd_jvp_max_relerr", maxA])
        w.writerow(["B_ratio_median", bridge["ratio_median"]])
        w.writerow(["B_ratio_mean", bridge["ratio_mean"]])
        w.writerow(["B_ratio_std", bridge["ratio_std"]])
        w.writerow(["B_ratio_p10", bridge["ratio_p10"]])
        w.writerow(["B_ratio_p90", bridge["ratio_p90"]])
        w.writerow(["B_kendall_tau_vk_dk", bridge["kendall_tau_vk_dk"]])
        w.writerow(["B_kendall_tau_wkvk_dk", bridge["kendall_tau_wkvk_dk"]])
        w.writerow(["B_kendall_tau_renorm_secondary", bridge["kendall_tau_renorm_secondary"]])
        w.writerow(["C_sigma_1", svd["sigma_1"]])
        w.writerow(["C_svd_linear_dV", svd["svd_linear_dV"]])
        w.writerow(["C_svd_true_dV", svd["svd_true_dV"]])
        w.writerow(["C_rand_linear_dV_max", svd["rand_linear_dV_max"]])
        w.writerow(["C_rand_true_dV_max", svd["rand_true_dV_max"]])
        w.writerow(["C_rand_true_dV_mean", svd["rand_true_dV_mean"]])
        w.writerow(["C_margin_true_vs_randmax", svd["margin_true_vs_randmax"]])
        w.writerow(["C_margin_true_vs_randmean", svd["margin_true_vs_randmean"]])
        w.writerow(["VERDICT_operator_agnostic", int(verdict)])
    print(f"[csv ] wrote {out_csv}")

    # per-edge bridge detail CSV (top by d_k)
    out_edges = ROOT / "scripts" / "results_universal_rl_edges.csv"
    bridge_rows.sort(key=lambda x: x["d_k"], reverse=True)
    with open(out_edges, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["i", "j", "w_k", "v_k",
                                             "wk_vk", "d_k", "d_k_renorm",
                                             "ratio"])
        wcsv.writeheader()
        for row in bridge_rows:
            wcsv.writerow(row)
    print(f"[csv ] wrote {out_edges}")

    return {
        "verdict": verdict, "rho": rho, "fp_res": fp_res,
        "fd_max": maxA, "fd_median": medA,
        "ratio_median": bridge["ratio_median"], "tau": bridge["kendall_tau_vk_dk"],
        "svd_margin_true": svd["margin_true_vs_randmax"],
        "n_edges": len(edges),
    }


if __name__ == "__main__":
    main()
