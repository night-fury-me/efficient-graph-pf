"""PIVOTAL reachability experiment — can a budget-bounded WORST-CASE attack
drive a trained implicit GNN to adversarial CRITICALITY rho(J_z) -> 1?

Go/no-go for the AEGIS breakthrough bid. The old phase experiment used the
*wrong* attack: it perturbed along v1 = leading right singular vector of S_c,
which maximizes the FIRST-ORDER EQUILIBRIUM SHIFT sigma_1(S_c) (a resolvent /
non-normality direction), NOT the spectral radius rho(J_z). This script
implements the CORRECT critical-driving attack (PGD that maximizes rho(J_z)
under Ahat + delta) and contrasts it against the v1 baseline and a random
baseline, exactly at the same Frobenius budget.

Threat model (matched to scripts/exp_phase_transition.py EXACTLY):
  - perturbation delta applied DIRECTLY to the normalized adjacency, NO
    renormalization: A_pert = A_sub + delta, ctx_pert = {**ctx_sub, "A_hat": A_pert}
  - delta symmetric, supported on A_sub's nonzero edges (same support v1 uses)
  - ||delta||_F <= eps

Linearized driving (theory-aligned, differentiable). The operator is
F(Z) = relu(Ahat @ (Z @ W^T) + Xproj). Its state Jacobian is
J_z = diag(phi') (Ahat (x) W) where phi' is the relu mask at the equilibrium
pre-activation. We FREEZE phi' at the CLEAN equilibrium mask and recompute only
the Ahat-dependence under Ahat + delta. This is exactly the memo's
J_z'(eps) = diag(phi') (Ahat'(eps) (x) W), whose all-active spectral law is
rho(J_z) = rho(Ahat) rho(W) and whose breaking budget is
eps* = 1/rho(W) - rho(Ahat) via the rank-1 sign-aligned top-eigenvector
construction delta* = eps * v_top v_top^T (Crux memo C1).

Definitions (Crux memo paper/review/breakthrough_crux_C1C4.md):
  eps_crit = (1 - kappa)/||W||_2          (norm budget; phase-exp convention)
  eps_star = 1/rho(W) - rho(Ahat)         (spectral breaking budget, all-active)
  eta      = ||(I-J_z)^{-1}||_2 * (1 - rho(J_z))   (non-normality / pseudospectral)
  g        = 1 - rho(J_z)                 (order parameter; vanishes at eps_star)
  eps_reach = smallest eps (interpolated) with rho(J_z under critical-driving) >= 1

RUN: toy self-check (a) + ONE smoke (kappa0=0.9, seed 42, coarse eps grid).
Full sweep gated behind --full (do NOT run until code is critiqued).

Usage:
    .venv/bin/python scripts/exp_reachability.py            # toy + smoke
    .venv/bin/python scripts/exp_reachability.py --full     # full sweep + CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (  # noqa: E402
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.ift import compute_jacobian  # noqa: E402

# Reuse the EXACT training / model / data plumbing from the phase experiment so
# the trained IGNN, equilibrium, and J_z are identical to the old result.
from exp_phase_transition import (  # noqa: E402
    IGNN_Kappa,
    set_seed,
    train_ignn_kappa,
)
from iem.examples.ignn_cora import _load_cora  # noqa: E402


# ---------------------------------------------------------------------------
# Edge-support bookkeeping (the v1 baseline's support: upper-triangular nonzero
# entries of A_sub, mirrored symmetrically). Matches run_single's edge_list loop.
# ---------------------------------------------------------------------------
def edge_support(A_sub: Tensor, include_diag: bool = False) -> list[tuple[int, int]]:
    """Unique upper-triangular nonzero edges (i<j) of A_sub. The real threat
    model perturbs off-diagonal edges only (matches run_single's edge_list).

    include_diag=True additionally appends nonzero diagonal (i,i) entries; used
    ONLY by the toy self-check so the rank-1 optimum delta*=v v^T (which has a
    diagonal) is feasible and the analytic law is attainable.
    """
    N = A_sub.shape[0]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_sub[i, j].abs() > 1e-10:
                edges.append((i, j))
    if include_diag:
        for i in range(N):
            if A_sub[i, i].abs() > 1e-10:
                edges.append((i, i))
    return edges


def edges_to_delta(weights: Tensor, edges: list[tuple[int, int]], N: int,
                   device, dtype) -> Tensor:
    """Scatter a per-edge weight vector into a symmetric NxN delta (diag entries
    untouched; same as run_single's dA[i,j]=dA[j,i]=w[k])."""
    dA = torch.zeros(N, N, device=device, dtype=dtype)
    for k, (i, j) in enumerate(edges):
        dA[i, j] = weights[k]
        dA[j, i] = weights[k]
    return dA


def delta_to_edges(dA: Tensor, edges: list[tuple[int, int]]) -> Tensor:
    """Inverse: read the upper-triangular edge weights out of a symmetric dA."""
    return torch.stack([dA[i, j] for (i, j) in edges])


def scale_to_fro(dA: Tensor, eps: float) -> Tensor:
    """Scale dA to EXACT Frobenius norm eps (boundary projection). eps=0 returns
    the ZERO matrix; a degenerate (all-zero) dA stays zero."""
    if eps == 0.0:
        return torch.zeros_like(dA)
    fro = float(dA.norm())
    if fro < 1e-30:
        return dA
    return dA * (eps / fro)


# ---------------------------------------------------------------------------
# Linearized driving Jacobian J_z(Ahat + delta) with phi' FROZEN at the clean
# equilibrium. Built by the SAME row-by-row autograd convention as
# iem.ift.compute_jacobian, so the vec ordering matches exactly.
# ---------------------------------------------------------------------------
def make_frozen_mask(model: IGNN_Kappa, Z_star: Tensor, ctx: dict) -> Tensor:
    """relu'(pre-activation) at the clean equilibrium, as a 0/1 mask (N,hidden).

    Pre-activation P* = Ahat @ (Z* @ W^T) + Xproj. relu'(P*) = 1{P*>0}.
    Frozen across the whole attack (linearized driving).
    """
    A_hat = ctx["A_hat"]
    X_proj = ctx["X_proj"]
    with torch.no_grad():
        pre = A_hat @ model.W(Z_star) + X_proj
        mask = (pre > 0).to(Z_star.dtype)
    return mask


def jz_under(model, Z_star, X_proj, mask, A_pert) -> Tensor:
    """J_z of the FROZEN-phi' linear operator F(Z)=mask*(A_pert @ Z @ W^T + Xproj)
    evaluated at the clean equilibrium, built DIRECTLY via the Kronecker form

        J_z = diag(vec(mask)) @ kron(A_pert, W)

    where vec is ROW-MAJOR over (N, hidden) — the EXACT vec convention that
    iem.ift.compute_jacobian uses (verified bit-identical, max err 0.0). This is
    fully differentiable in A_pert (hence in delta) and O(D^2), not the O(D^3*D)
    autograd row loop. Returns (D, D) with D = N*hidden.

    Mask/X_proj are constants here (frozen phi'); X_proj drops out of the state
    Jacobian. Z_star is unused (the linearized operator is affine in Z).
    """
    W = model.W.weight  # (hidden, hidden); operator uses W(Z)=Z @ W.weight^T
    # row-major (N,h): J[i*h+c, j*h+d] = mask[i,c] * A_pert[i,j] * W[c,d]
    K = torch.kron(A_pert, W)                       # (N*h, N*h)
    return mask.reshape(-1).unsqueeze(1) * K        # diag(vec mask) @ K


# ---------------------------------------------------------------------------
# Differentiable dominant-eigenvalue magnitude rho_hat(J_z), with a
# power-iteration Rayleigh fallback for (near-)defective matrices.
# ---------------------------------------------------------------------------
def rho_hat_eig(J: Tensor) -> Tensor:
    """Primary path: |lambda_max| via torch.linalg.eigvals (gradients supported
    for non-defective J)."""
    return torch.linalg.eigvals(J).abs().max()


# cache a fixed init vector per D so the power-iteration graph is deterministic
_POWER_V0: dict[int, Tensor] = {}


def _power_v0(D: int, device, dtype) -> Tensor:
    key = D
    v = _POWER_V0.get(key)
    if v is None:
        g = torch.Generator(device="cpu").manual_seed(1234)
        v = torch.randn(D, generator=g, dtype=torch.double)
        v = v / v.norm()
        _POWER_V0[key] = v
    return v.to(device, dtype)


def rho_hat_power(J: Tensor, iters: int = 60) -> Tensor:
    """Differentiable dominant-eigenvalue MAGNITUDE via fixed-step power
    iteration on J, with a Rayleigh-quotient readout. Robust on (near-)defective
    matrices where eigvals' backward (which solves with the eigenvector matrix)
    fails. Fixed init + fixed step count => deterministic autograd graph.

    For a real dominant eigenpair the Rayleigh quotient rho ~ |v^T J v|; we also
    take |J v| (= magnitude scaling per step) and return the max of the two,
    which tracks |lambda_max| for both real-dominant and complex-pair-dominant J.
    This is the GRADIENT path only; the reported rho always uses exact eigvals
    (rho_eval).
    """
    D = J.shape[0]
    v = _power_v0(D, J.device, J.dtype)
    last_scale = J.new_tensor(0.0)
    for _ in range(iters):
        w = J @ v
        nw = w.norm()
        if float(nw.detach()) < 1e-30:
            return J.new_tensor(0.0).abs()
        last_scale = nw          # |J v| with ||v||=1  (per-step magnitude)
        v = w / nw
    rayleigh = (v @ (J @ v)).abs()          # |v^T J v|
    return torch.maximum(rayleigh, last_scale)


def differentiable_rho(J: Tensor, use_power: bool) -> tuple[Tensor, str]:
    """Pick the differentiable-rho path. Returns (rho, path_label).

    For large D (>= 400) the eigvals backward is both slow (~1.6s) and fragile
    on near-defective J, so we use the power-iteration Rayleigh path for the
    GRADIENT by default. Exact rho for reporting is always taken via eigvals.
    """
    if use_power or J.shape[0] >= 400:
        return rho_hat_power(J), "power"
    try:
        r = rho_hat_eig(J)
        if torch.isnan(r) or torch.isinf(r):
            raise RuntimeError("nan/inf from eigvals")
        return r, "eig"
    except Exception:
        return rho_hat_power(J), "power"


def rho_eval(J: Tensor) -> float:
    """Non-differentiable, robust EVALUATION of rho(J) for reporting (uses
    eigvals; this is the ground-truth spectral radius). The GPU cusolverDnXgeev
    backend can throw CUSOLVER_STATUS_INTERNAL_ERROR on some matrices (hit on
    seeds 3141/9999); fall back to CPU LAPACK, which is reliable and numerically
    identical (the spectral radius is backend-independent)."""
    with torch.no_grad():
        try:
            return float(torch.linalg.eigvals(J).abs().max())
        except RuntimeError:
            return float(torch.linalg.eigvals(J.detach().cpu()).abs().max())


# ---------------------------------------------------------------------------
# The critical-driving PGD attack (the core).
# ---------------------------------------------------------------------------
def _rank1_warmstart_edges(model, A_sub, edges, eps):
    """Analytic warm start: project the rank-1 ideal delta* = v_top v_top^T
    (top eigenvector of A_sub, sign-matched to W's dominant eigenvalue) onto the
    edge support, return the per-edge weight vector (NOT yet scaled to eps).

    On a fully-connected symmetric support this IS the global rho-optimum
    (Crux memo C1). On a sparse support it is the best rank-1 seed for PGD.
    """
    device, dtype = A_sub.device, A_sub.dtype
    evals, evecs = torch.linalg.eigh(A_sub)
    k = int(torch.argmax(evals.abs()))
    vtop = evecs[:, k]
    # sign so the rank-1 bump ADDS to rho along the dominant W direction:
    # |.|-based rho is sign-agnostic for the bump magnitude, but matching the
    # sign of lambda_max(A_sub) keeps the construction aligned.
    if float(evals[k]) < 0:
        vtop = vtop  # magnitude unchanged; |lambda| is what matters
    dA = vtop.unsqueeze(1) @ vtop.unsqueeze(0)       # (N,N) rank-1, symmetric
    return delta_to_edges(dA, edges)                  # restrict to support


def pgd_critical_attack(
    model, Z_star, X_proj, mask, A_sub, edges, eps,
    iters=120, lrs=(0.3, 1.0), verbose=False,
):
    """Projected gradient ASCENT maximizing rho(J_z(A_sub + delta)) subject to
    delta symmetric, edge-supported, ||delta||_F = eps.

    Uses TANGENT-SPACE projected ascent on the Frobenius sphere (remove the
    radial component of the gradient before stepping, then renormalize to the
    boundary) — proper PGD on a sphere, which converges to the rank-1 optimum
    far better than naive ascend-then-rescale. The analytic rank-1 warm start
    (top eigvec of A_sub on the edge support) is ALSO entered as a candidate, so
    the attack never under-performs the closed-form construction.

    Returns dict: best delta (NxN), best rho_hat (ground-truth rho), path used.
    """
    N = A_sub.shape[0]
    device, dtype = A_sub.device, A_sub.dtype
    n_edges = len(edges)

    if eps == 0.0 or n_edges == 0:
        z = torch.zeros(N, N, device=device, dtype=dtype)
        J0 = jz_under(model, Z_star, X_proj, mask, A_sub)
        return {"delta": z, "rho_hat": rho_eval(J0), "lr": 0.0, "path": "none",
                "fro": 0.0}

    def eval_edges(e_vec):
        """Ground-truth rho(J_z) for an edge-weight vector scaled to eps."""
        dA = edges_to_delta(e_vec, edges, N, device, dtype)
        dA = scale_to_fro(dA, eps)
        return rho_eval(jz_under(model, Z_star, X_proj, mask, A_sub + dA)), dA

    # Decide rho-grad path ONCE. For large D the eigvals backward is slow and
    # fragile, so force the power path (differentiable_rho also enforces this);
    # the probe below only runs for small D to catch defective-J instability.
    D = mask.numel()
    use_power = D >= 400
    if not use_power:
        try:
            e_probe = torch.randn(n_edges, generator=torch.Generator().manual_seed(7),
                                  dtype=torch.double).to(device, dtype)
            e_probe.requires_grad_(True)
            dA_probe = edges_to_delta(e_probe, edges, N, device, dtype)
            Jp = jz_under(model, Z_star, X_proj, mask, A_sub + 1e-3 * dA_probe)
            rp = rho_hat_eig(Jp)
            (gp,) = torch.autograd.grad(rp, e_probe, retain_graph=False)
            if torch.isnan(rp) or torch.isinf(rp) or torch.isnan(gp).any() or torch.isinf(gp).any():
                use_power = True
        except Exception:
            use_power = True
    path_label = "power" if use_power else "eig"

    # candidate seeds: analytic rank-1 warm start + one random direction
    seeds = [_rank1_warmstart_edges(model, A_sub, edges, eps),
             torch.randn(n_edges, generator=torch.Generator().manual_seed(99),
                         dtype=torch.double).to(device, dtype)]

    best = {"delta": None, "rho_hat": -1.0, "lr": None, "path": path_label,
            "fro": 0.0}

    # seed the best with the analytic warm start itself (closed-form floor)
    for s in seeds:
        if float(s.norm()) > 1e-30:
            r0, dA0 = eval_edges(s)
            if r0 > best["rho_hat"]:
                best = {"delta": dA0.detach(), "rho_hat": r0, "lr": "warmstart",
                        "path": path_label, "fro": float(dA0.norm())}

    for seed_vec in seeds:
        if float(seed_vec.norm()) < 1e-30:
            continue
        for lr in lrs:
            # start on the sphere boundary
            dA = scale_to_fro(edges_to_delta(seed_vec, edges, N, device, dtype), eps)
            e = delta_to_edges(dA, edges).detach()
            for it in range(iters):
                e.requires_grad_(True)
                dA = edges_to_delta(e, edges, N, device, dtype)
                J = jz_under(model, Z_star, X_proj, mask, A_sub + dA)
                rho, _ = differentiable_rho(J, use_power)
                (grad,) = torch.autograd.grad(rho, e, retain_graph=False)
                with torch.no_grad():
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        if not use_power:        # switch to power for the rest
                            use_power = True
                            path_label = "power"
                        e = e.detach()
                        continue
                    e = e.detach()
                    enorm = e.norm()
                    if float(enorm) > 1e-30:
                        # TANGENT-SPACE step: remove radial (along e) component
                        u = e / enorm
                        grad_tan = grad - (grad @ u) * u
                        e = e + lr * grad_tan
                    else:
                        e = e + lr * grad
                    # renormalize edge vector so ||dA||_F = eps (boundary)
                    dA = edges_to_delta(e, edges, N, device, dtype)
                    dA = scale_to_fro(dA, eps)
                    e = delta_to_edges(dA, edges).detach()
            r_fin, dA_fin = eval_edges(e)
            if r_fin > best["rho_hat"]:
                best = {"delta": dA_fin.detach(), "rho_hat": r_fin, "lr": lr,
                        "path": path_label, "fro": float(dA_fin.norm())}

    return best


# ---------------------------------------------------------------------------
# Baselines.
# ---------------------------------------------------------------------------
def v1_delta(model, Z_star, ctx_sub, A_sub, edges, eps):
    """v1 baseline = eps * leading right singular vector of S_c, scattered onto
    edges (EXACTLY as run_single), then scaled to ||.||_F = eps."""
    N = A_sub.shape[0]
    device, dtype = A_sub.device, A_sub.dtype
    J_z_struct, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_star, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_star, ctx_sub,
        J_z=J_z_struct, J_A=J_A,
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return torch.zeros(N, N, device=device, dtype=dtype)
    _, _, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
    weights = Vh_c[0]
    dA = torch.zeros(N, N, device=device, dtype=dtype)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = weights[k]
        dA[j, i] = weights[k]
    return scale_to_fro(dA, eps)


def random_delta(A_sub, edges, eps, seed):
    """Random symmetric edge-supported perturbation scaled to ||.||_F = eps."""
    N = A_sub.shape[0]
    device, dtype = A_sub.device, A_sub.dtype
    g = torch.Generator().manual_seed(2024 + seed)
    w = torch.randn(len(edges), generator=g, dtype=torch.double).to(device, dtype)
    dA = edges_to_delta(w, edges, N, device, dtype)
    return scale_to_fro(dA, eps)


# ---------------------------------------------------------------------------
# Full nonlinear reconvergence (phase-exp divergence guard) under A_sub+delta.
# ---------------------------------------------------------------------------
def reconverge_full(model, Z_init, ctx_pert, max_iter=300, tol=1e-8):
    """Reconverge the TRUE nonlinear equilibrium (relu active). Returns
    (Z, diverged, rho_recon). Matches run_single's guard: NaN or norm>1e6."""
    Z = Z_init.clone()
    diverged = False
    with torch.no_grad():
        Z_new = Z
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_pert)
            if torch.isnan(Z_new).any() or float(Z_new.norm()) > 1e6:
                diverged = True
                break
            if (Z_new - Z).norm() < tol * max(float(Z.norm()), 1.0):
                Z = Z_new
                break
            Z = Z_new
    if diverged:
        return Z_new, True, float("inf")
    # rho of the TRUE J_z at the reconverged point (relu mask recomputed there)
    def F_z(z):
        return model.operator(z.reshape(Z.shape), ctx_pert).reshape(-1)
    try:
        J = compute_jacobian(F_z, Z)
        rho_recon = rho_eval(J)
    except Exception:
        rho_recon = float("nan")
    return Z, False, rho_recon


def predictions(model, Z) -> Tensor:
    with torch.no_grad():
        return model.head(Z).argmax(dim=1)


def resolvent_norm(J: Tensor) -> float:
    J = J.detach()
    D = J.shape[0]
    I = torch.eye(D, device=J.device, dtype=J.dtype)
    try:
        return float(torch.linalg.svdvals(torch.linalg.inv(I - J))[0])
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Memo quantities computed from the trained model on the subgraph.
# ---------------------------------------------------------------------------
def memo_quantities(model, A_sub, kappa) -> dict:
    """eps_crit (norm), eps_star (spectral), rho(Ahat), rho(W), ||W||, ||Ahat||."""
    with torch.no_grad():
        W = model.W.weight.detach()
        W_sn = float(torch.linalg.svdvals(W)[0])
        rho_W = float(torch.linalg.eigvals(W).abs().max())
        A_sn = float(torch.linalg.svdvals(A_sub)[0])
        rho_A = float(torch.linalg.eigvals(A_sub).abs().max())
    eps_crit = (1.0 - kappa) / W_sn if W_sn > 1e-10 else float("inf")
    eps_star = (1.0 / rho_W - rho_A) if rho_W > 1e-10 else float("inf")
    return {"eps_crit": eps_crit, "eps_star": eps_star, "W_sn": W_sn,
            "rho_W": rho_W, "A_sn": A_sn, "rho_A": rho_A}


# ---------------------------------------------------------------------------
# Per-(kappa0) machinery: train, build subgraph + equilibrium + frozen mask.
# ---------------------------------------------------------------------------
def prepare(data, device, seed, kappa):
    """Train IGNN, build 50-node subgraph, reconverge equilibrium, frozen mask,
    clean J_z. Mirrors run_single up to J_z."""
    set_seed(seed)
    model, Z_star, ctx, A_hat_sn = train_ignn_kappa(data, device, seed, kappa)
    A_hat = data["A_hat"].to(device)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}
    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        Z_new = Z_sub
        for _ in range(300):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    mask = make_frozen_mask(model, Z_sub, ctx_sub)
    edges = edge_support(A_sub)

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    J_z_clean = compute_jacobian(F_z, Z_sub)
    rho_clean = rho_eval(J_z_clean)

    return {
        "model": model, "A_sub": A_sub, "X_proj_sub": X_proj_sub,
        "ctx_sub": ctx_sub, "Z_sub": Z_sub, "mask": mask, "edges": edges,
        "J_z_clean": J_z_clean, "rho_clean": rho_clean,
        "pred_clean": predictions(model, Z_sub),
    }


# ---------------------------------------------------------------------------
# Evaluate all three attacks at a single eps.
# ---------------------------------------------------------------------------
def eval_eps(prep, mem, eps, seed, pgd_iters=200):
    model = prep["model"]; A_sub = prep["A_sub"]; mask = prep["mask"]
    X_proj = prep["X_proj_sub"]; Z_sub = prep["Z_sub"]; edges = prep["edges"]
    ctx_sub = prep["ctx_sub"]
    device, dtype = A_sub.device, A_sub.dtype
    N = A_sub.shape[0]

    rows = []

    # --- critical-driving (the core) ---
    atk = pgd_critical_attack(model, Z_sub, X_proj, mask, A_sub, edges, eps,
                              iters=pgd_iters)
    dA = atk["delta"]
    J_drive = jz_under(model, Z_sub, X_proj, mask, A_sub + dA)
    rho_drive = rho_eval(J_drive)
    res_drive = resolvent_norm(J_drive)
    # FULL nonlinear reconvergence under the driving perturbation
    ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
    Z_rec, diverged, rho_recon = reconverge_full(model, Z_sub, ctx_pert)
    flips = int((predictions(model, Z_rec) != prep["pred_clean"]).sum()) if not diverged else N
    rows.append({"attack": "critical_driving", "rho_jz": rho_drive,
                 "resolvent_norm": res_drive, "diverged": int(diverged),
                 "flips": flips, "rho_recon": rho_recon, "fro": atk["fro"],
                 "path": atk["path"]})

    # --- v1 baseline ---
    dA1 = v1_delta(model, Z_sub, ctx_sub, A_sub, edges, eps)
    J_v1 = jz_under(model, Z_sub, X_proj, mask, A_sub + dA1)
    rows.append({"attack": "v1", "rho_jz": rho_eval(J_v1),
                 "resolvent_norm": resolvent_norm(J_v1), "diverged": 0,
                 "flips": -1, "rho_recon": float("nan"), "fro": float(dA1.norm()),
                 "path": "n/a"})

    # --- random baseline ---
    dAr = random_delta(A_sub, edges, eps, seed)
    J_r = jz_under(model, Z_sub, X_proj, mask, A_sub + dAr)
    rows.append({"attack": "random", "rho_jz": rho_eval(J_r),
                 "resolvent_norm": resolvent_norm(J_r), "diverged": 0,
                 "flips": -1, "rho_recon": float("nan"), "fro": float(dAr.norm()),
                 "path": "n/a"})

    for r in rows:
        r.update({"eps": eps, "eps_crit": mem["eps_crit"],
                  "eps_star": mem["eps_star"]})
    return rows


def eta_at(J: Tensor, rho: float) -> float:
    rn = resolvent_norm(J)
    naive = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")
    return rn / naive if naive < float("inf") else float("inf")


# ===========================================================================
# SELF-CHECK (a): toy sanity on a small matrix with KNOWN rho-optimum.
# ===========================================================================
def toy_self_check() -> bool:
    """Build a small symmetric Ahat (4 nodes path-ish) and a fixed W. With phi'=1
    and a symmetric J_z = Ahat (x) W, the all-active law gives
    rho(J_z(Ahat+delta)) = rho(Ahat+delta) * rho(W). The rho-MAXIMIZING feasible
    delta on the FULL support (||delta||_F = eps) is the rank-1 sign-aligned
    top-eigenvector construction delta* = eps * v_top v_top^T, which raises
    rho(Ahat) by exactly eps (its spectral norm). We confirm PGD on the
    linearized J_z increases rho and approaches rho(W)*(rho(Ahat)+eps).

    We use a SEPARATE, self-contained linear operator here (not the IGNN) so the
    analytic optimum is exact. delta is restricted to the edge support of Ahat;
    we make Ahat fully connected (all off-diagonals nonzero) so the rank-1
    optimum lies inside the feasible set and the known optimum is attainable.
    """
    print("\n" + "=" * 70)
    print("  SELF-CHECK (a): toy rho-maximization vs analytic optimum")
    print("=" * 70)
    torch.set_default_dtype(torch.double)
    device = torch.device("cpu")
    dtype = torch.double
    N = 4

    # symmetric Ahat with ALL off-diagonal edges present (full support), small
    # diagonal; normalize spectral radius to a known value.
    g = torch.Generator().manual_seed(11)
    M = torch.randn(N, N, generator=g, dtype=dtype)
    A0 = (M + M.T) / 2
    A0 = A0 - torch.diag(torch.diag(A0)) + 0.1 * torch.eye(N)  # keep diag small
    rho_A0 = float(torch.linalg.eigvalsh(A0).abs().max())
    A0 = A0 / rho_A0 * 0.6  # rho(Ahat) = 0.6
    rho_A = float(torch.linalg.eigvalsh(A0).abs().max())

    # symmetric W with rho(W) = 0.9 (so rho(J_z)=rho(A)rho(W)=0.54 clean)
    Wg = torch.randn(3, 3, generator=g, dtype=dtype)
    W = (Wg + Wg.T) / 2
    rho_W = float(torch.linalg.eigvalsh(W).abs().max())
    W = W / rho_W * 0.9
    rho_W = float(torch.linalg.eigvalsh(W).abs().max())

    # frozen-phi'=1 linear operator: F(Z) = A_hat @ (Z @ W) + 0  (W symmetric)
    # Build a tiny "model-like" closure compatible with jz_under by emulating
    # model.W as a linear map and mask=1, X_proj=0, Z* arbitrary (J_z is
    # Z-independent for a linear op, so any Z* works).
    class _ToyW:
        def __init__(self, Wmat):
            self.weight = Wmat
        def __call__(self, Z):
            return Z @ self.weight.T

    class _ToyModel:
        def __init__(self, Wmat):
            self.W = _ToyW(Wmat)

    toy = _ToyModel(W)
    Z_star = torch.zeros(N, 3, dtype=dtype)
    X_proj = torch.zeros(N, 3, dtype=dtype)
    mask = torch.ones(N, 3, dtype=dtype)
    # FULL support INCLUDING the diagonal, so the rank-1 optimum delta*=v v^T
    # (which has diagonal mass) is feasible and the analytic law is attainable.
    full = torch.ones(N, N, dtype=dtype)
    edges = edge_support(full, include_diag=True)

    # clean rho
    J0 = jz_under(toy, Z_star, X_proj, mask, A0)
    rho0 = rho_eval(J0)
    print(f"  rho(Ahat)={rho_A:.4f}  rho(W)={rho_W:.4f}  "
          f"clean rho(J_z)={rho0:.4f}  (expect ~{rho_A*rho_W:.4f})")

    eps = 0.25
    # ANALYTIC optimum: delta* = eps * v_top v_top^T (top eigvec of Ahat),
    # raising rho(Ahat) to rho_A + eps; rho(J_z*) = (rho_A+eps)*rho_W.
    evals, evecs = torch.linalg.eigh(A0)
    k = int(torch.argmax(evals.abs()))
    vtop = evecs[:, k:k + 1]
    sign = 1.0 if float(evals[k]) >= 0 else 1.0  # rho uses |.|; rank-1 adds +eps
    dA_opt = sign * eps * (vtop @ vtop.T)
    dA_opt = scale_to_fro(dA_opt, eps)
    J_opt = jz_under(toy, Z_star, X_proj, mask, A0 + dA_opt)
    rho_opt_analytic = rho_eval(J_opt)
    rho_known = (rho_A + eps) * rho_W
    print(f"  analytic delta* rho(J_z)={rho_opt_analytic:.4f}  "
          f"known (rho_A+eps)*rho_W={rho_known:.4f}")

    # PGD attack
    atk = pgd_critical_attack(toy, Z_star, X_proj, mask, A0, edges, eps,
                              iters=300, lrs=(1.0, 0.5, 2.0, 0.2))
    rho_pgd = atk["rho_hat"]
    print(f"  PGD rho(J_z)={rho_pgd:.4f}  (path={atk['path']}, "
          f"||delta||_F={atk['fro']:.4f})")

    ok_increase = rho_pgd > rho0 + 1e-3
    ok_optimum = rho_pgd >= 0.97 * rho_opt_analytic  # within 3% of analytic opt
    ok_known = abs(rho_opt_analytic - rho_known) < 1e-6
    print(f"  CHECK increase(rho_pgd>rho0): {ok_increase}")
    print(f"  CHECK approaches optimum (>=97% analytic): {ok_optimum} "
          f"[{rho_pgd:.4f} vs {rho_opt_analytic:.4f}]")
    print(f"  CHECK analytic==known law: {ok_known}")
    passed = ok_increase and ok_optimum and ok_known
    print(f"  TOY SELF-CHECK: {'PASS' if passed else 'FAIL'}")
    torch.set_default_dtype(torch.float)
    return passed


# ===========================================================================
# Smoke: kappa0=0.9, seed 42, coarse eps grid, all three attacks.
# ===========================================================================
def smoke(data, device, kappa0=0.9, seed=42, n_grid=6):
    print("\n" + "=" * 70)
    print(f"  SMOKE: kappa0={kappa0}, seed={seed}, coarse grid")
    print("=" * 70, flush=True)
    prep = prepare(data, device, seed, kappa0)
    mem = memo_quantities(prep["model"], prep["A_sub"], kappa0)
    print(f"  rho(Ahat)={mem['rho_A']:.4f} ||Ahat||={mem['A_sn']:.4f}  "
          f"rho(W)={mem['rho_W']:.4f} ||W||={mem['W_sn']:.4f}")
    print(f"  clean rho(J_z)={prep['rho_clean']:.4f}  "
          f"eps_crit={mem['eps_crit']:.4f}  eps_star={mem['eps_star']:.4f}", flush=True)

    # Span up to the larger of 3*eps_crit and 2*eps_star. Here eps_star (the
    # TRUE spectral breaking budget) is ~eta x larger than eps_crit (the norm
    # budget), and the masked/non-normal realized crossing sits near ~1.5-2x
    # eps_star, so a 3*eps_crit-only grid (the old phase-exp range) cannot reach
    # rho=1 even when the system IS reachable — that under-range is itself a
    # finding (eps_crit drastically under-estimates reachability).
    eps_max = max(3.0 * mem["eps_crit"], 2.0 * mem["eps_star"])
    eps_grid = np.linspace(0.0, eps_max, n_grid)
    print(f"\n  eps grid (0 -> {eps_max:.4f} = max(3*eps_crit, 2*eps_star)): "
          f"{', '.join(f'{e:.4f}' for e in eps_grid)}")
    print(f"\n  {'eps':>8} {'eps/ec':>7} | {'rho_drive':>10} {'rho_v1':>8} "
          f"{'rho_rand':>9} | {'div':>3} {'flips':>5} {'rho_recon':>9} {'path':>6}")
    print("  " + "-" * 78)

    drive_curve = []  # (eps, rho_drive)
    for eps in eps_grid:
        rows = eval_eps(prep, mem, float(eps), seed, pgd_iters=200)
        rmap = {r["attack"]: r for r in rows}
        d = rmap["critical_driving"]
        drive_curve.append((float(eps), d["rho_jz"]))
        print(f"  {eps:8.4f} {eps/mem['eps_crit']:7.2f} | "
              f"{d['rho_jz']:10.4f} {rmap['v1']['rho_jz']:8.4f} "
              f"{rmap['random']['rho_jz']:9.4f} | {d['diverged']:3d} "
              f"{d['flips']:5d} {d['rho_recon'] if math.isfinite(d['rho_recon']) else float('inf'):9.4f} "
              f"{d['path']:>6}", flush=True)

    # discrimination check at the largest eps
    eps_last = float(eps_grid[-1])
    rows = eval_eps(prep, mem, eps_last, seed, pgd_iters=200)
    rmap = {r["attack"]: r for r in rows}
    rd, rv = rmap["critical_driving"]["rho_jz"], rmap["v1"]["rho_jz"]
    print(f"\n  DISCRIMINATION at eps={eps_last:.4f}: "
          f"rho_drive={rd:.4f} vs rho_v1={rv:.4f}")
    if rd > rv + 0.05:
        print(f"  -> critical-driving dominates v1 by {rd - rv:.4f} (EXPECTED).")
    else:
        print(f"  -> !!! v1 matches/beats driving (gap {rd - rv:+.4f}) — SURPRISE, "
              f"investigate before trusting.")

    eps_reach = interp_eps_reach(drive_curve)
    verdict(eps_reach, mem)
    return drive_curve, mem


def interp_eps_reach(curve):
    """Smallest eps where rho_drive >= 1, linearly interpolated between grid pts.
    Returns None if rho never reaches 1 on the grid."""
    for k in range(1, len(curve)):
        e0, r0 = curve[k - 1]
        e1, r1 = curve[k]
        if r0 < 1.0 <= r1:
            if r1 == r0:
                return e1
            return e0 + (1.0 - r0) * (e1 - e0) / (r1 - r0)
    if curve and curve[0][1] >= 1.0:
        return curve[0][0]
    return None


def verdict(eps_reach, mem):
    print("\n  " + "=" * 40)
    if eps_reach is None:
        print("  VERDICT: NOT-REACHABLE on this eps grid "
              "(rho(J_z) under critical-driving < 1 throughout).")
    else:
        ratio_c = eps_reach / mem["eps_crit"] if mem["eps_crit"] > 0 else float("inf")
        rel_star = eps_reach / mem["eps_star"] if mem["eps_star"] > 0 else float("inf")
        print(f"  VERDICT: REACHABLE. eps_reach={eps_reach:.4f}")
        print(f"           eps_reach/eps_crit={ratio_c:.3f}  "
              f"eps_reach/eps_star={rel_star:.3f}")
        print(f"           (theory: eps_reach should track eps_star={mem['eps_star']:.4f}, "
              f"not eps_crit={mem['eps_crit']:.4f})")
    print("  " + "=" * 40, flush=True)


# ===========================================================================
# FULL sweep (gated; writes CSV).
# ===========================================================================
def full_sweep(data, device, seed=42, n_grid=16):
    out = Path("results/exp_reachability.csv")
    out.parent.mkdir(exist_ok=True)
    fieldnames = ["kappa0", "eps", "attack", "rho_jz", "resolvent_norm",
                  "diverged", "flips", "eps_crit", "eps_star", "eta",
                  "rho_recon", "fro", "path"]
    all_rows = []
    summary = {}
    for kappa0 in (0.5, 0.9):
        print("\n" + "#" * 70)
        print(f"#  kappa0 = {kappa0}")
        print("#" * 70, flush=True)
        prep = prepare(data, device, seed, kappa0)
        mem = memo_quantities(prep["model"], prep["A_sub"], kappa0)
        eta_clean = eta_at(prep["J_z_clean"], prep["rho_clean"])
        eps_max = max(3.0 * mem["eps_crit"], 2.0 * mem["eps_star"])
        eps_grid = np.linspace(0.0, eps_max, n_grid)
        drive_curve = []
        for eps in eps_grid:
            rows = eval_eps(prep, mem, float(eps), seed, pgd_iters=250)
            for r in rows:
                J = None  # eta only meaningful for the driving J_z; reuse rho<1
                eta_val = (r["resolvent_norm"] * (1.0 - r["rho_jz"])
                           if r["rho_jz"] < 1.0 else float("inf"))
                all_rows.append({
                    "kappa0": kappa0, "eps": r["eps"], "attack": r["attack"],
                    "rho_jz": r["rho_jz"], "resolvent_norm": r["resolvent_norm"],
                    "diverged": r["diverged"], "flips": r["flips"],
                    "eps_crit": r["eps_crit"], "eps_star": r["eps_star"],
                    "eta": eta_val, "rho_recon": r["rho_recon"],
                    "fro": r["fro"], "path": r["path"],
                })
                if r["attack"] == "critical_driving":
                    drive_curve.append((r["eps"], r["rho_jz"]))
        eps_reach = interp_eps_reach(drive_curve)
        summary[kappa0] = {"mem": mem, "eta_clean": eta_clean,
                           "eps_reach": eps_reach, "curve": drive_curve}
        verdict(eps_reach, mem)

    with out.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        for r in all_rows:
            wtr.writerow(r)
    print(f"\nWrote {len(all_rows)} rows -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="run full kappa0 sweep + write CSV (gated)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # SELF-CHECK (a) ALWAYS runs first.
    toy_ok = toy_self_check()
    if not toy_ok:
        print("\n!!! TOY SELF-CHECK FAILED — attack is buggy; do NOT trust "
              "reachability results. Aborting before model runs.", flush=True)
        if not args.full:
            return
        # still abort full sweep on a failed toy check
        return

    data = _load_cora(Path("datasets/cora"))
    print(f"\nCora: N={data['N']}, features={data['n_features']}, "
          f"classes={data['n_classes']}")

    if args.full:
        full_sweep(data, device, seed=args.seed)
    else:
        smoke(data, device, kappa0=0.9, seed=args.seed, n_grid=6)


if __name__ == "__main__":
    main()
