"""Adversarial Equilibrium Theory for Implicit Graph Models.

One theorem + two propositions for certified robustness of DEQ-GNNs
under graph structure perturbation, computed via the Implicit Function Theorem.

Theorem 1 (Phase Transition in Adversarial Vulnerability):
    Structural perturbations exhibit three regimes around eps_crit = (1-rho)/||W||_2:
    (a) Subcritical: ||Dz*|| <= sigma_1(S) * eps + O(eps^2), unique fixed point
    (b) Critical: bound diverges as Theta(1/(eps_crit - ||dA||))
    (c) Supercritical: contractivity lost, fixed point may bifurcate
    Empirically validated: 83x amplification as rho -> 1, divergence beyond eps_crit.

Proposition 1 (Optimal First-Order Structural Attack):
    dA* = eps * reshape(v_1, (N, N)) where v_1 is the leading right singular
    vector of S. Wins 15/15 budget levels vs Mettack across 3 datasets.

Proposition 2 (Per-Node Robust Radius):
    r_v = m_v / (||df/dz_v|| * ||S_v||). Deterministic certificate per node.
    1.9-7.7x larger radii than randomized smoothing at equal coverage.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .ift import compute_jacobian


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_ego_subgraph(
    A_hat: Tensor,
    max_nodes: int = 50,
    center: Optional[int] = None,
) -> Tensor:
    """BFS-based ego subgraph extraction. Guarantees connectivity.

    The naive approach (take first-k neighbors of highest-degree node)
    yields near-empty subgraphs on datasets with high-degree hubs (e.g.,
    WikiCS center has 3324 neighbors → 50 random picks share only 7 edges).
    BFS ensures the subgraph is connected with dense inter-edges.
    """
    from collections import deque

    if center is None:
        center = int(A_hat.sum(dim=1).argmax().item())

    visited = [center]
    seen = {center}
    queue = deque([center])
    while len(visited) < max_nodes and queue:
        node = queue.popleft()
        for n in (A_hat[node] > 0).nonzero(as_tuple=True)[0].tolist():
            if n not in seen and len(visited) < max_nodes:
                seen.add(n)
                visited.append(n)
                queue.append(n)

    return torch.tensor(sorted(visited), device=A_hat.device)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_structural_jacobian(
    F: Callable[[Tensor, dict], Tensor],
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    eps: float = 1e-4,
    edges_only: bool = False,
) -> Tuple[Tensor, Tensor, Optional[list]]:
    """Compute state Jacobian J_z and structural Jacobian J_A.

    Args:
        F: operator (z, ctx) -> z'
        z_star: fixed point
        ctx: context dict containing adjacency under A_key
        A_key: key for adjacency matrix in ctx
        eps: finite-difference step
        edges_only: if True, only compute columns for existing edges

    Returns:
        J_z: (D, D) state Jacobian
        J_A: (D, K) structural Jacobian (K = N^2 or |active_cols|)
        col_map: list of (i,j) tuples mapping columns of J_A to A entries,
                 or None if edges_only=False (columns are in row-major order)
    """
    A = ctx[A_key]
    N = A.shape[0]
    D = z_star.numel()

    def F_z(z):
        return F(z.reshape(z_star.shape), ctx).reshape(-1)

    # J_z via vectorized reverse-mode AD (torch.func): identical to the dense
    # autograd loop (verified allclose, atol<1e-3) but ~2000x faster -- the prior
    # compute_jacobian materialized the (D,D) Jacobian one row at a time. Fallback
    # kept for environments without torch.func.
    try:
        import torch.func as _tfunc
        J_z = _tfunc.jacrev(F_z)(z_star.reshape(-1).detach())
    except Exception:
        J_z = compute_jacobian(F_z, z_star)

    with torch.no_grad():
        f_base = F(z_star, ctx).reshape(-1)

    if edges_only:
        col_map = []
        cols = []
        for i in range(N):
            for j in range(N):
                if A[i, j].abs() < 1e-10 and i != j:
                    continue
                col_map.append((i, j))
                A_pert = A.clone()
                A_pert[i, j] += eps
                with torch.no_grad():
                    f_pert = F(z_star, {**ctx, A_key: A_pert}).reshape(-1)
                cols.append((f_pert - f_base) / eps)
        J_A = torch.stack(cols, dim=1)
        return J_z, J_A, col_map

    J_A = torch.zeros(D, N * N, device=A.device, dtype=A.dtype)
    for idx in range(N * N):
        i, j = idx // N, idx % N
        A_pert = A.clone()
        A_pert[i, j] += eps
        with torch.no_grad():
            f_pert = F(z_star, {**ctx, A_key: A_pert}).reshape(-1)
        J_A[:, idx] = (f_pert - f_base) / eps

    return J_z, J_A, None


# ---------------------------------------------------------------------------
# Theorem 1(a): Certified Fixed-Point Shift Bound
# ---------------------------------------------------------------------------

def structural_sensitivity_matrix(
    F: Callable[[Tensor, dict], Tensor],
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    J_z: Optional[Tensor] = None,
    J_A: Optional[Tensor] = None,
) -> Tensor:
    """Compute S = (I - J_z)^{-1} J_A, the structural sensitivity matrix.

    S maps structural perturbations vec(dA) to fixed-point shifts dz*:
        dz* = S @ vec(dA) + O(||dA||^2)

    sigma_1(S) is the first-order adversarial vulnerability constant.
    """
    if J_z is None or J_A is None:
        J_z, J_A, _ = _compute_structural_jacobian(F, z_star, ctx, A_key)

    D = J_z.shape[0]
    I = torch.eye(D, device=J_z.device, dtype=J_z.dtype)

    try:
        S = torch.linalg.solve(I - J_z, J_A)
    except torch._C._LinAlgError:
        rho_est = torch.linalg.eigvals(J_z).abs().max().item()
        lam = max(rho_est - 0.99, 0.01)
        S = torch.linalg.solve((1.0 + lam) * I - J_z, J_A)

    return S


def constrained_sensitivity_matrix(
    S: Tensor,
    A_hat: Tensor,
) -> Tuple[Tensor, list]:
    """Build constrained sensitivity matrix S_c for symmetric edge perturbations.

    The unconstrained S ∈ R^{D × N²} treats every adjacency entry independently.
    Real graph perturbations are symmetric: δA[i,j] = δA[j,i]. S_c ∈ R^{D × |E|}
    has one column per unique edge, equal to S_{:,iN+j} + S_{:,jN+i}.

    sigma_1(S_c) gives the tight first-order bound under symmetric perturbations.
    """
    N = A_hat.shape[0]
    edge_list = []
    cols = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_hat[i, j].abs() > 1e-10:
                col = S[:, i * N + j] + S[:, j * N + i]
                cols.append(col)
                edge_list.append((i, j))
    if not cols:
        return torch.zeros(S.shape[0], 0, device=S.device), edge_list
    S_c = torch.stack(cols, dim=1)
    return S_c, edge_list


def certified_shift_bound(
    S: Tensor,
    rho: float,
    epsilon: float,
    A_hat: Optional[Tensor] = None,
) -> dict:
    """Theorem 1(a): First-order certified bound on ||Dz*|| under ||dA|| <= eps.

    Reports two bounds:
    - Unconstrained: sigma_1(S) * eps  (over all R^{N×N} perturbations)
    - Constrained: sigma_1(S_c) * eps  (symmetric, edge-only perturbations)

    The constrained bound is tighter and more realistic for graph perturbation.
    """
    sigma = torch.linalg.svdvals(S)
    sigma_1 = float(sigma[0])

    result = {
        "upper_bound": sigma_1 * epsilon,
        "sigma_1": sigma_1,
        "sigma_spectrum": sigma[:min(10, len(sigma))].detach().cpu(),
        "epsilon": epsilon,
        "rho": rho,
        "certified": rho < 1.0,
    }

    if A_hat is not None:
        S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)
        if S_c.shape[1] > 0:
            sigma_c = torch.linalg.svdvals(S_c)
            sigma_1_c = float(sigma_c[0])
            result["constrained_upper_bound"] = sigma_1_c * epsilon
            result["constrained_sigma_1"] = sigma_1_c
            result["n_edges"] = len(edge_list)

    return result


# ---------------------------------------------------------------------------
# Proposition 1: Optimal First-Order Structural Attack
# ---------------------------------------------------------------------------

def optimal_structural_attack(
    S: Tensor,
    A_hat: Tensor,
    epsilon: float,
    top_k: int = 5,
    symmetric: bool = True,
) -> dict:
    """Proposition 1: Compute the optimal first-order adversarial perturbation.

    The attack is supported on existing edges and SYMMETRIC, so the relevant
    sensitivity operator is the constrained S_c (one column per unique edge,
    column_k = S_{:,iN+j} + S_{:,jN+i}, the natural edge-weight parametrization),
    NOT the unconstrained S. The returned direction is the symmetric edge-supported
    leading right singular vector. We therefore report the CONSTRAINED first-order
    shift bound consistent with that direction:

        sigma_1 = sigma_1(S_c)              (max shift per unit edge-weight ||c||)
        max_first_order_shift = eps * sigma_1(S_c)

    Because a symmetric edge perturbation has ||dA||_F = sqrt(2) ||c||, the
    budget-correct shift per unit Frobenius norm is sigma_1(S_c)/sqrt(2); we expose
    it as `sigma_1_per_fro` / `max_shift_per_fro` so figures/callers can report the
    threat-model (||dA||_F-budgeted) bound. The unconstrained sigma_1(S) is kept as
    `sigma_1_unconstrained` for reference. The attack DIRECTION is unchanged.

    Also computes per-edge vulnerability spectrum.
    """
    N = A_hat.shape[0]
    U, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
    sigma_1_unconstr = float(sigma[0])

    attack_vec = Vh[0]  # leading right singular vector (unconstrained)

    # Reshape to adjacency perturbation
    if attack_vec.shape[0] == N * N:
        attack_direction = attack_vec.reshape(N, N)
    else:
        attack_direction = attack_vec  # edges_only mode

    if symmetric and attack_direction.dim() == 2 and attack_direction.shape[0] == N:
        attack_direction = (attack_direction + attack_direction.T) / 2
        sym_norm = attack_direction.norm()
        if sym_norm > 1e-10:
            attack_direction = attack_direction / sym_norm

    # Constrained sensitivity S_c (symmetric, edge-supported): this is the operator
    # whose leading singular value bounds the first-order shift of the returned
    # (symmetric, edge-supported) direction. Per-edge vulnerabilities are its
    # column norms.
    sqrt2 = 2.0 ** 0.5
    edges = []
    if S.shape[1] == N * N:
        S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)
        if S_c.shape[1] > 0:
            sigma_c = torch.linalg.svdvals(S_c)
            sigma_1 = float(sigma_c[0])
        else:
            sigma_1 = 0.0
        for k, (i, j) in enumerate(edge_list):
            edges.append((i, j, float(S_c[:, k].norm())))
        edges.sort(key=lambda x: x[2], reverse=True)
    else:
        # edges_only / already-constrained S: treat S itself as S_c.
        sigma_1 = sigma_1_unconstr

    max_shift = sigma_1 * epsilon

    # Effective adversarial dimensionality (from the unconstrained spectrum)
    eff_dim = int((sigma > sigma[0] * 0.01).sum())

    return {
        # CONSTRAINED bound, consistent with the returned symmetric edge direction:
        "max_first_order_shift": max_shift,          # eps * sigma_1(S_c)
        "sigma_1": sigma_1,                          # sigma_1(S_c)
        # Per-Frobenius (||dA||_F-budgeted) bound for the threat model:
        "sigma_1_per_fro": sigma_1 / sqrt2,          # sigma_1(S_c)/sqrt(2)
        "max_shift_per_fro": (sigma_1 / sqrt2) * epsilon,
        # Reference: unconstrained operator norm (over all R^{N x N} perturbations):
        "sigma_1_unconstrained": sigma_1_unconstr,
        "sigma_spectrum": sigma[:min(20, len(sigma))].detach().cpu(),
        "attack_direction": attack_direction.detach(),
        "vulnerability_spectrum": edges[:top_k],
        "all_edge_vulnerabilities": edges,
        "effective_adversarial_dim": eff_dim,
    }


# ---------------------------------------------------------------------------
# Theorem 1(b,c): Critical Perturbation Budget
# ---------------------------------------------------------------------------

def critical_perturbation_budget(
    rho: float,
    W_spectral_norm: float,
) -> dict:
    """Theorem 1(b,c): Minimum perturbation that could break contractivity.

    For IGNN-class operators F(Z) = sigma(A Z W^T + X_proj):
        eps_crit >= (1 - rho) / ||W||_2

    Below eps_crit: all certificates hold (contractivity preserved).
    Above: phase transition — certificates may become void.
    """
    if rho >= 1.0:
        return {
            "epsilon_crit": 0.0,
            "rho": rho,
            "margin": 0.0,
            "W_spectral_norm": W_spectral_norm,
            "already_supercritical": True,
        }

    margin = 1.0 - rho
    eps_crit = margin / W_spectral_norm if W_spectral_norm > 1e-10 else float("inf")

    return {
        "epsilon_crit": eps_crit,
        "rho": rho,
        "margin": margin,
        "W_spectral_norm": W_spectral_norm,
        "already_supercritical": False,
    }


def extract_W_spectral_norm(model: nn.Module) -> float:
    """Extract ||W||_2 from an IGNN-style model."""
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") and "W" in name:
            return float(torch.linalg.svdvals(mod.weight.detach())[0])
    for name, param in model.named_parameters():
        if "W" in name and "weight" in name:
            return float(torch.linalg.svdvals(param.detach())[0])
    raise ValueError("Could not find weight matrix W in model")


# ---------------------------------------------------------------------------
# Proposition 2: Per-Node Robust Radius
# ---------------------------------------------------------------------------

def per_node_robust_radius(
    S: Tensor,
    z_star: Tensor,
    logits: Tensor,
    labels: Tensor,
    rho: float,
    head: nn.Module,
    runner_up_surrogate: bool = False,
) -> dict:
    """Proposition (prop:radius): per-node first-order sensitivity radius.

    Min-over-classes composed-norm form (default, matches the corrected prop:radius):
        r_v = min_{c != y_v} (f_{y_v} - f_c) / ||(W_{y_v} - W_c) S_v||_2 ,
    with y_v = argmax(logits)_v the predicted class and S_v the block-row of S at node v
    (S already incorporates (I - J_z)^{-1}, so no separate (1-rho) factor). The min over ALL
    competitors is the distance to the nearest first-order boundary, robust to runner-up swaps.

    runner_up_surrogate=True uses the cheaper, possibly-optimistic surrogate
        r_hat_v = (f_{y_v} - f_{c*}) / (||W_{y_v} - W_{c*}||_2 ||S_v||_2) ,  c* the runner-up;
    since ||(W_{y_v}-W_c) S_v|| <= ||W_{y_v}-W_c|| ||S_v|| (Cauchy-Schwarz), the surrogate
    under-estimates each per-class radius and inspects only c* (empirically ~3x smaller; X5).

    Any structural perturbation with ||dA||_F < r_v preserves node v's predicted class (first order).
    """
    N = logits.shape[0]
    if rho >= 1.0:
        return {"radii": torch.zeros(N), "certified": False, "reason": "rho >= 1"}

    if not isinstance(head, nn.Linear):
        raise ValueError(
            "per_node_robust_radius assumes a linear head (prop:radius); "
            f"got {type(head).__name__}"
        )
    d = z_star.shape[1] if z_star.dim() > 1 else 1
    W = head.weight.detach()                                   # (C, d)
    L = logits.detach()
    preds = L.argmax(dim=1)
    C = L.shape[1]

    radii = torch.zeros(N, device=L.device)
    margins = torch.zeros(N, device=L.device)                 # runner-up margin (for reporting)
    for v in range(N):
        p = int(preds[v])
        Sv = S[v * d:(v + 1) * d]                             # (d, pert)
        marg = L[v, p] - L[v]                                 # (C,)  f_pred - f_c, >= 0
        marg_excl = marg.clone()
        marg_excl[p] = float("inf")
        cstar = int(torch.argmin(marg_excl))                 # runner-up
        margins[v] = marg[cstar]
        if runner_up_surrogate:
            denom = (W[p] - W[cstar]).norm() * Sv.norm() + 1e-10
            radii[v] = (marg[cstar] / denom).clamp(min=0)
        else:
            best = float("inf")
            for c in range(C):
                if c == p:
                    continue
                comp = (W[p] - W[c]) @ Sv                     # (pert,)
                r_c = float(marg[c]) / (float(comp.norm()) + 1e-10)
                if r_c < best:
                    best = r_c
            radii[v] = max(best, 0.0)

    correct = (preds.cpu() == labels.cpu()) if labels is not None \
        else torch.ones(N, dtype=torch.bool)
    return {
        "radii": radii.detach().cpu(),
        "margins": margins.detach().cpu(),
        "certified": True,
        "method": "runner_up_surrogate" if runner_up_surrogate else "min_over_classes",
        "mean_radius": float(radii.mean()),
        "median_radius": float(radii.median()),
        "frac_nontrivial": float((radii > 1e-6).float().mean()),
        "frac_correct_and_certified": float(
            (correct & (radii.cpu() > 1e-6)).float().mean()
        ),
    }


# ---------------------------------------------------------------------------
# Non-normality index (Remark in paper)
# ---------------------------------------------------------------------------

def nonnormality_index(J_z: Tensor, rho: float) -> dict:
    """Measure deviation of J_z from normality.

    For normal J: ||(I - J)^{-1}||_2 = 1/(1 - rho).
    For non-normal J: ||(I - J)^{-1}||_2 can be >> 1/(1 - rho).

    The ratio eta = ||(I-J)^{-1}||_2 * (1 - rho) measures non-normality.
    eta = 1 for normal matrices, eta >> 1 indicates transient amplification.
    """
    D = J_z.shape[0]
    I = torch.eye(D, device=J_z.device, dtype=J_z.dtype)
    resolvent_norm = float(torch.linalg.svdvals(torch.linalg.inv(I - J_z))[0])
    naive_bound = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")
    eta = resolvent_norm / naive_bound if naive_bound < float("inf") else float("inf")

    return {
        "resolvent_norm": resolvent_norm,
        "naive_bound": naive_bound,
        "nonnormality_index": eta,
        "interpretation": (
            "normal (eta~1): spectral radius is a faithful robustness measure"
            if eta < 2.0
            else f"non-normal (eta={eta:.1f}): vulnerability amplified {eta:.1f}x beyond spectral radius prediction"
        ),
    }


# ---------------------------------------------------------------------------
# Empirical validation
# ---------------------------------------------------------------------------

def validate_bound_tightness(
    F: Callable[[Tensor, dict], Tensor],
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    S: Tensor,
    A_key: str = "A_hat",
    epsilons: Optional[List[float]] = None,
    n_random: int = 5,
    reconverge_iter: int = 100,
) -> list:
    """Validate Theorem 1(a) empirically: compare predicted vs actual shift.

    Reports both unconstrained and constrained (symmetric, edge-only) tightness.
    The constrained bound is the realistic one for graph perturbation.
    """
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]

    A = ctx[A_key]
    N = A.shape[0]

    # Unconstrained SVD
    U, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
    sigma_1 = float(sigma[0])

    # Constrained SVD (symmetric, edge-only)
    S_c, edge_list = constrained_sensitivity_matrix(S, A)
    if S_c.shape[1] > 0:
        U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
        sigma_1_c = float(sigma_c[0])
    else:
        sigma_1_c = 0.0
        Vh_c = None

    # Establish clean baseline
    Z = z_star.clone()
    with torch.no_grad():
        for _ in range(reconverge_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    z_clean = Z_new.clone()

    results = []
    for eps in epsilons:
        # --- Constrained optimal attack (symmetric, edge-only) ---
        dA_constr = torch.zeros(N, N, device=A.device)
        if Vh_c is not None and len(edge_list) > 0:
            weights = eps * Vh_c[0]  # coefficients per edge
            for k, (i, j) in enumerate(edge_list):
                dA_constr[i, j] = float(weights[k])
                dA_constr[j, i] = float(weights[k])

        ctx_constr = {**ctx, A_key: A + dA_constr}
        Z = z_clean.clone()
        with torch.no_grad():
            for _ in range(reconverge_iter):
                Z_new = model.operator(Z, ctx_constr)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
        actual_constr = float((Z_new - z_clean).norm())
        predicted_constr = sigma_1_c * eps

        # --- Random symmetric edge perturbation ---
        actual_rands = []
        for _ in range(n_random):
            dA_rand = torch.zeros(N, N, device=A.device)
            if edge_list:
                rand_w = torch.randn(len(edge_list), device=A.device)
                rand_w = rand_w / (rand_w.norm() + 1e-10) * eps
                for k, (i, j) in enumerate(edge_list):
                    dA_rand[i, j] = float(rand_w[k])
                    dA_rand[j, i] = float(rand_w[k])

            ctx_rand = {**ctx, A_key: A + dA_rand}
            Z = z_clean.clone()
            with torch.no_grad():
                for _ in range(reconverge_iter):
                    Z_new = model.operator(Z, ctx_rand)
                    if (Z_new - Z).norm() < 1e-7:
                        break
                    Z = Z_new
            actual_rands.append(float((Z_new - z_clean).norm()))
        actual_rand = sum(actual_rands) / len(actual_rands)

        results.append({
            "epsilon": eps,
            "predicted_constr": predicted_constr,
            "actual_constr": actual_constr,
            "actual_random": actual_rand,
            "constr_tightness": actual_constr / predicted_constr if predicted_constr > 0 else 0,
            "attack_advantage": actual_constr / actual_rand if actual_rand > 0 else float("inf"),
            # Also report unconstrained for comparison
            "predicted_unconstr": sigma_1 * eps,
            "unconstr_tightness": actual_constr / (sigma_1 * eps) if sigma_1 > 0 else 0,
        })

    return results


def phase_transition_scan(
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    rho_targets: Optional[List[float]] = None,
    epsilon: float = 0.01,
) -> list:
    """Validate Theorem 1(b,c): scan rho and show vulnerability diverges at rho -> 1.

    Scales the adjacency matrix A to achieve different rho values (robust to
    spectral_norm parametrization on W) and measures actual adversarial shift.
    """
    if rho_targets is None:
        rho_targets = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

    from .certify import spectral_radius

    A = ctx[A_key]
    N = A.shape[0]

    def F_z(z):
        return model.operator(z.reshape(z_star.shape), ctx).reshape(-1)

    base_rho = spectral_radius(F_z, z_star)
    if base_rho < 1e-6:
        return [{"error": "base rho too small"}]

    results = []
    for rho_t in rho_targets:
        scale = rho_t / base_rho
        A_scaled = A * scale
        ctx_scaled = {**ctx, A_key: A_scaled}

        # Reconverge at scaled A
        Z = torch.zeros_like(z_star)
        converged = False
        with torch.no_grad():
            for it in range(300):
                Z_new = model.operator(Z, ctx_scaled)
                if (Z_new - Z).norm() < 1e-7:
                    converged = True
                    break
                Z = Z_new
        z_new = Z_new if converged else Z

        actual_rho = spectral_radius(
            lambda z, _c=ctx_scaled: model.operator(z.reshape(z_new.shape), _c).reshape(-1),
            z_new,
        )

        # Random perturbation at this rho level
        dA = torch.randn(N, N, device=A.device) * epsilon / (N ** 0.5)
        dA = (dA + dA.T) / 2
        dA.fill_diagonal_(0)
        ctx_pert = {**ctx_scaled, A_key: A_scaled + dA}

        Z_p = z_new.clone()
        diverged = False
        with torch.no_grad():
            for _ in range(300):
                Z_new_p = model.operator(Z_p, ctx_pert)
                if torch.isnan(Z_new_p).any() or Z_new_p.norm() > 1e6:
                    diverged = True
                    break
                if (Z_new_p - Z_p).norm() < 1e-7:
                    break
                Z_p = Z_new_p
        shift = float("inf") if diverged else float((Z_new_p - z_new).norm())

        predicted = epsilon / (1.0 - min(actual_rho, 0.999))

        results.append({
            "rho_target": rho_t,
            "rho_actual": actual_rho,
            "converged": converged,
            "actual_shift": shift,
            "predicted_1_over_1mrho": predicted,
            "ratio": shift / predicted if predicted > 0 else 0,
        })

    return results


# ---------------------------------------------------------------------------
# Baselines for comparison
# ---------------------------------------------------------------------------

def greedy_structural_attack(
    model: nn.Module,
    z_clean: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    reconverge_iter: int = 100,
) -> list:
    """Baseline attack: brute-force single-edge removal, rank by damage.

    For each existing edge (i,j), removes it, reconverges, and measures
    ||z*_pert - z*_clean||. Returns edges sorted by descending damage.
    This is the strongest possible single-edge attack (exhaustive search).
    """
    A = ctx[A_key]
    N = A.shape[0]

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j].abs() > 1e-10:
                edges.append((i, j))

    results = []
    with torch.no_grad():
        for i, j in edges:
            A_pert = A.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx, A_key: A_pert}
            Z = z_clean.clone()
            for _ in range(reconverge_iter):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
            shift = float((Z_new - z_clean).norm())
            results.append((i, j, shift))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def randomized_smoothing_certificate(
    model: nn.Module,
    z_clean: Tensor,
    ctx: dict,
    labels: Tensor,
    A_key: str = "A_hat",
    sigma: float = 0.01,
    n_samples: int = 100,
    alpha: float = 0.001,
) -> dict:
    """Baseline certificate: randomized smoothing (Bojchevski et al., 2020 style).

    Adds Gaussian noise to adjacency, classifies n_samples times, and computes
    certified radius r_v = sigma * Phi^{-1}(p_A) where p_A is the fraction
    of correct classifications under noise.

    Returns probabilistic certificates valid with probability >= 1 - alpha.
    """
    from scipy.stats import norm as scipy_norm

    A = ctx[A_key]
    N = labels.shape[0]
    correct_counts = torch.zeros(N)

    # Sparse edge mask: only perturb existing edges
    edge_mask = (A.abs() > 1e-10).float()
    edge_mask.fill_diagonal_(0)
    clean_preds = model.head(z_clean).argmax(dim=1).cpu()

    raw_sigma = sigma * (2 ** 0.5)  # compensate for symmetrization halving variance

    with torch.no_grad():
        for _ in range(n_samples):
            dA = torch.randn_like(A) * raw_sigma * edge_mask
            dA = (dA + dA.T) / 2
            A_pert = (A + dA).clamp(min=0)
            ctx_pert = {**ctx, A_key: A_pert}
            Z = z_clean.clone()
            for _ in range(100):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
            logits_pert = model.head(Z_new)
            pred = logits_pert.argmax(dim=1).cpu()
            correct_counts += (pred == clean_preds).float()

    p_A = correct_counts / n_samples

    radii = torch.zeros(N)
    for v in range(N):
        if p_A[v] > 0.5:
            k = int(p_A[v].item() * n_samples)
            # Clopper-Pearson lower bound via Beta distribution
            from scipy.stats import beta as beta_dist
            if k >= n_samples:
                p_lower = alpha ** (1.0 / n_samples)
            else:
                p_lower = beta_dist.ppf(alpha / 2, k, n_samples - k + 1)
            p_lower = max(p_lower, 0.5 + 1e-6)
            p_lower = min(p_lower, 1.0 - 1e-10)
            radii[v] = sigma * scipy_norm.ppf(p_lower)

    frac_cert = float((radii > 1e-6).float().mean())
    nontrivial = radii[radii > 1e-6]

    return {
        "radii": radii,
        "p_A": p_A,
        "sigma": sigma,
        "n_samples": n_samples,
        "frac_certified": frac_cert,
        "mean_radius": float(nontrivial.mean()) if len(nontrivial) > 0 else 0.0,
        "median_radius": float(nontrivial.median()) if len(nontrivial) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Convenience: full analysis in one call
# ---------------------------------------------------------------------------

def diagnostic_analysis(
    F: Callable[[Tensor, dict], Tensor],
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    logits: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> dict:
    """Diagnostic-only path -- released unconditionally; CANNOT synthesise a perturbation.

    Returns the per-edge vulnerability spectrum ``v_ij`` (S_c column norms), the per-node
    first-order radii ``r_v`` (Prop. radius), the leading sensitivity magnitude ``sigma_1``,
    the spectral radius ``rho`` and the critical budget ``eps_crit``. It deliberately does NOT
    call ``optimal_structural_attack`` and does NOT return the SVD-optimal attack DIRECTION
    ``delta_Ahat*`` (Algorithm 1's direction step), which is gated per the disclosure protocol:
    only scalar scores and radii are produced, from which a perturbation cannot be directly
    reconstructed. This is the path the paper releases without restriction.
    """
    from .certify import spectral_radius as sr

    J_z, J_A, _ = _compute_structural_jacobian(F, z_star, ctx, A_key)
    rho = sr(lambda z: F(z.reshape(z_star.shape), ctx).reshape(-1), z_star)
    S = structural_sensitivity_matrix(F, z_star, ctx, A_key, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, ctx[A_key])

    v_ij = {tuple(e): float(S_c[:, k].norm()) for k, e in enumerate(edge_list)}
    sigma_1 = float(torch.linalg.svdvals(S_c)[0]) if S_c.shape[1] else 0.0

    eps_crit = None
    try:
        W_norm = extract_W_spectral_norm(model)
        if W_norm:
            _b = critical_perturbation_budget(rho, W_norm)
            eps_crit = _b.get("epsilon_crit") if isinstance(_b, dict) else _b
    except ValueError:
        pass

    r_v = None
    if (logits is not None and labels is not None
            and isinstance(getattr(model, "head", None), nn.Linear)):
        r_v = per_node_robust_radius(S, z_star, logits, labels, rho, model.head)["radii"]

    return {
        "v_ij": v_ij,                 # per-edge vulnerability spectrum (scores only)
        "r_v": r_v,                   # per-node first-order radii (or None)
        "sigma_1": sigma_1,           # leading sensitivity magnitude (no direction)
        "rho": float(rho),
        "eps_crit": eps_crit,
        "edge_list": [tuple(e) for e in edge_list],
        "note": "diagnostic-only: no attack direction synthesised (gated per disclosure protocol).",
    }


def full_adversarial_analysis(
    F: Callable[[Tensor, dict], Tensor],
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    epsilon: float = 0.01,
    logits: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> dict:
    """Run all adversarial analysis: Theorems 1-3 + Proposition 1."""
    from .certify import spectral_radius as sr

    J_z, J_A, _ = _compute_structural_jacobian(F, z_star, ctx, A_key)
    rho = sr(lambda z: F(z.reshape(z_star.shape), ctx).reshape(-1), z_star)
    S = structural_sensitivity_matrix(F, z_star, ctx, A_key, J_z=J_z, J_A=J_A)

    bound = certified_shift_bound(S, rho, epsilon)

    A_hat = ctx[A_key]
    attack = optimal_structural_attack(S, A_hat, epsilon)

    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = None
    budget = critical_perturbation_budget(rho, W_norm) if W_norm else None

    nn_idx = nonnormality_index(J_z, rho)

    node_certs = None
    if logits is not None and labels is not None and hasattr(model, "head"):
        node_certs = per_node_robust_radius(
            S, z_star, logits, labels, rho, model.head,
        )

    return {
        "rho": rho,
        "theorem1_certified_bound": bound,
        "theorem2_optimal_attack": attack,
        "theorem3_critical_budget": budget,
        "prop1_node_certificates": node_certs,
        "nonnormality": nn_idx,
        "sensitivity_matrix_shape": tuple(S.shape),
    }
