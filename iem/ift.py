"""Implicit Function Theorem (IFT) sensitivity computation.

Core primitive for IEM. Given a contractive fixed-point operator F and
its equilibrium z* = F(z*), computes:

    ∂z*/∂p = (I - ∂F/∂z)⁻¹ · ∂F/∂p

where p is any differentiable parameter (edge weight, node feature, etc.).

Two solvers:
  - Direct: full Jacobian + torch.linalg.solve (exact, O(D³), for D ≤ ~500)
  - Neumann: iterative J-vector products (approximate, O(K·D), for large D)

The Neumann series v = Σ_{k=0}^{K} J^k · b converges when ρ(J) < 1
(guaranteed by contractivity).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def compute_jacobian(
    F: Callable[[Tensor], Tensor],
    z: Tensor,
) -> Tensor:
    """Full Jacobian ∂F/∂z via manual row-by-row backward.

    torch.autograd.functional.jacobian (even vectorize=False) silently
    produces wrong results for operators with complex-number ops. Manual
    row-by-row backward is verified correct against finite differences.

    Args:
        F: operator z -> z' (must be differentiable)
        z: point at which to evaluate (D,) or (B, N, C) flattened internally

    Returns:
        J: (D, D) Jacobian matrix
    """
    D = z.numel()
    rows = []
    with torch.enable_grad():
        for i in range(D):
            z_in = z.detach().clone().reshape(-1).requires_grad_(True)
            f_out = F(z_in.reshape(z.shape)).reshape(-1)
            grad = torch.autograd.grad(f_out[i], z_in, retain_graph=False)[0]
            rows.append(grad)
    return torch.stack(rows)


def ift_solve_direct(
    F: Callable[[Tensor], Tensor],
    z_star: Tensor,
    dF_dp: Tensor,
) -> Tensor:
    """Solve (I - J) · v = dF_dp via direct linear solve.

    Args:
        F: fixed-point operator
        z_star: converged equilibrium (any shape, flattened internally)
        dF_dp: ∂F/∂p at z_star, shape (D,) or (D, P) for P parameters

    Returns:
        v: ∂z*/∂p, same shape as dF_dp
    """
    D = z_star.numel()
    J = compute_jacobian(F, z_star)  # (D, D)
    I = torch.eye(D, device=J.device, dtype=J.dtype)
    A = I - J  # (D, D)

    b = dF_dp.reshape(D, -1) if dF_dp.dim() > 1 else dF_dp.reshape(D, 1)
    try:
        v = torch.linalg.solve(A, b)
    except torch._C._LinAlgError:
        # (I - J) is singular when ρ(J) ≥ 1. Fall back to ridge-regularized
        # least-squares: solve ((1+λ)I - J) · v = b with λ chosen adaptively
        # so the system is well-conditioned. This is the standard remedy for
        # near-singular IFT systems (cf. Lorraine+2020 implicit-MAML).
        rho_est = torch.linalg.eigvals(J).abs().max().item()
        lam = max(rho_est - 0.99, 0.01)  # push effective ρ below 1
        A_reg = (1.0 + lam) * I - J
        v = torch.linalg.solve(A_reg, b)

    return v.reshape(dF_dp.shape)


def ift_solve_neumann(
    F: Callable[[Tensor], Tensor],
    z_star: Tensor,
    dF_dp: Tensor,
    n_terms: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    """Solve (I - J)⁻¹ · dF_dp via Neumann series using JVPs.

    v = (I + J + J² + ...) · b  where b = dF_dp.
    Each term requires one JVP (forward-mode AD through F).
    Converges geometrically when ρ(J) < 1.

    Args:
        F: fixed-point operator
        z_star: converged equilibrium
        dF_dp: ∂F/∂p, shape matching z_star (flattened internally)
        n_terms: max Neumann terms
        tol: early stop when ||J^k · b|| < tol

    Returns:
        v: ∂z*/∂p approximation
    """
    z_flat = z_star.detach().reshape(-1)
    b = dF_dp.detach().reshape(-1)

    def jvp(v_in: Tensor) -> Tensor:
        """Compute J · v via forward-mode AD (true JVP, not VJP)."""
        try:
            from torch.func import jvp as torch_jvp
            def F_flat(z):
                return F(z.reshape(z_star.shape)).reshape(-1)
            _, Jv = torch_jvp(F_flat, (z_flat,), (v_in,))
            return Jv
        except (ImportError, RuntimeError):
            eps_fd = 1e-5
            f_plus = F((z_flat + eps_fd * v_in).reshape(z_star.shape)).reshape(-1)
            f_base = F(z_flat.reshape(z_star.shape)).reshape(-1)
            return (f_plus - f_base) / eps_fd

    result = b.clone()
    term = b.clone()
    for k in range(1, n_terms + 1):
        term = jvp(term)
        result = result + term
        if term.norm() < tol * b.norm():
            break

    return result.reshape(dF_dp.shape)


def param_sensitivity(
    F: Callable[[Tensor, dict], Tensor],
    z_star: Tensor,
    ctx: dict,
    param_key: str,
    method: str = "auto",
    **solver_kwargs,
) -> Tensor:
    """Compute ∂z*/∂p for parameter p = ctx[param_key] via IFT.

    The cross-Jacobian ∂F/∂p is (D_state, D_param). The IFT gives:
        ∂z*/∂p = (I - ∂F/∂z)⁻¹ · ∂F/∂p     shape (D_state, D_param)

    For per-element sensitivity, take the column norms of the result.

    Args:
        F: operator (z, ctx) -> z'
        z_star: converged equilibrium
        ctx: context dict
        param_key: key in ctx for the parameter tensor
        method: "direct", "neumann", or "auto"

    Returns:
        sensitivity: (D_param,) per-element L2 sensitivity norms
    """
    D = z_star.numel()
    if method == "auto":
        method = "direct" if D <= 500 else "neumann"

    param = ctx[param_key]
    P = param.numel()

    def F_z_only(z: Tensor) -> Tensor:
        return F(z.reshape(z_star.shape), ctx).reshape(-1)

    # Compute cross-Jacobian ∂F/∂p: shape (D_state, D_param)
    # Manual row-by-row backward — torch.autograd.functional.jacobian silently
    # produces zeros for operators containing complex-number ops (exp(1j*th),
    # conj, .real views) even with vectorize=False. The row-by-row approach
    # is verified correct against finite differences.
    rows = []
    with torch.enable_grad():
      for i in range(D):
        p_leaf = param.detach().clone().reshape(-1).requires_grad_(True)
        ctx_p = {**ctx, param_key: p_leaf.reshape(param.shape)}
        f_out = F(z_star.detach(), ctx_p).reshape(-1)
        grad = torch.autograd.grad(f_out[i], p_leaf, retain_graph=False)[0]
        rows.append(grad.reshape(-1))
    dF_dp = torch.stack(rows)  # (D_state, D_param)

    # Solve (I - J_zz) · V = dF_dp  where V is (D_state, D_param)
    if method == "direct":
        J_zz = compute_jacobian(F_z_only, z_star)  # (D, D)
        I = torch.eye(D, device=J_zz.device, dtype=J_zz.dtype)
        A = I - J_zz
        try:
            V = torch.linalg.solve(A, dF_dp)
        except torch._C._LinAlgError:
            rho_est = torch.linalg.eigvals(J_zz).abs().max().item()
            lam = max(rho_est - 0.99, 0.01)
            A_reg = (1.0 + lam) * I - J_zz
            V = torch.linalg.solve(A_reg, dF_dp)
    else:
        # Neumann: solve column-by-column
        cols = []
        for j in range(P):
            col = ift_solve_neumann(F_z_only, z_star, dF_dp[:, j], **solver_kwargs)
            cols.append(col)
        V = torch.stack(cols, dim=1)  # (D_state, D_param)

    # Per-element sensitivity = column norms of V
    per_elem = V.norm(dim=0)  # (D_param,)
    return per_elem.reshape(param.shape)


# Convenience aliases
edge_sensitivity = param_sensitivity
node_sensitivity = param_sensitivity
