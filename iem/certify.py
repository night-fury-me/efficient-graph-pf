"""Certification module: spectral radius and Lipschitz bounds.

Given a contractive operator F with fixed point z*, certifies:

  1. ρ(∂F/∂z) < 1 — contractivity verification
  2. ||∂z*/∂p|| ≤ ||∂F/∂p|| / (1 - ρ) — sensitivity bound
  3. Ranking stability under adversarial perturbation ε

Theorem 4 (Certified sensitivity bound):
  For L-Lipschitz F with contractivity ρ = ||∂F/∂z||_op < 1,
  the sensitivity is bounded: ||∂z*/∂p|| ≤ ||∂F/∂p||_op / (1 - ρ).
  Under adversarial perturbation ||δ|| ≤ ε, the ranking change is
  bounded by 2ε · ||∂F/∂p|| / (1 - ρ)².
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .ift import compute_jacobian


def spectral_radius(
    F: Callable[[Tensor], Tensor],
    z_star: Tensor,
    method: str = "auto",
    n_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Estimate ρ(∂F/∂z) at the fixed point z*.

    Args:
        F: operator z -> z'
        z_star: fixed point
        method: "power" (power iteration), "exact" (full eigendecomp), "auto"
        n_iter: max iterations for power method
        tol: convergence tolerance

    Returns:
        rho: spectral radius estimate (should be < 1 for contractive F)
    """
    D = z_star.numel()
    if method == "auto":
        method = "exact" if D <= 500 else "power"

    if method == "exact":
        J = compute_jacobian(F, z_star)
        eigenvalues = torch.linalg.eigvals(J)
        return float(eigenvalues.abs().max().item())

    # Power iteration for largest singular value (upper bound on spectral radius)
    z_flat = z_star.detach().reshape(-1)

    def jvp(v: Tensor) -> Tensor:
        with torch.enable_grad():
            z_in = z_flat.detach().requires_grad_(True)
            f_out = F(z_in.reshape(z_star.shape)).reshape(-1)
            return torch.autograd.grad(
                f_out, z_in, grad_outputs=v,
                create_graph=False, retain_graph=False,
            )[0]

    v = torch.randn(D, device=z_star.device)
    v = v / v.norm()
    sigma = 0.0

    for _ in range(n_iter):
        Jv = jvp(v)
        sigma_new = float(Jv.norm().item())
        v = Jv / (Jv.norm() + 1e-12)
        if abs(sigma_new - sigma) < tol:
            break
        sigma = sigma_new

    return sigma


def certified_sensitivity_bound(
    dF_dp_norm: float,
    rho: float,
) -> float:
    """Compute the certified upper bound on ||∂z*/∂p||.

    From Theorem 4: ||∂z*/∂p|| ≤ ||∂F/∂p|| / (1 - ρ).

    Args:
        dF_dp_norm: operator norm of ∂F/∂p
        rho: spectral radius of ∂F/∂z (must be < 1)

    Returns:
        bound: certified upper bound on sensitivity
    """
    if rho >= 1.0:
        return float("inf")
    return dF_dp_norm / (1.0 - rho)


def ranking_stability_bound(
    dF_dp_norm: float,
    rho: float,
    epsilon: float,
) -> float:
    """Maximum change in sensitivity ranking under adversarial perturbation ε.

    From Theorem 4 corollary: Δ_rank ≤ 2ε · ||∂F/∂p|| / (1 - ρ)².

    Args:
        dF_dp_norm: operator norm of ∂F/∂p
        rho: spectral radius (< 1)
        epsilon: adversarial budget ||δ|| ≤ ε

    Returns:
        delta_rank: maximum ranking perturbation
    """
    if rho >= 1.0:
        return float("inf")
    return 2.0 * epsilon * dF_dp_norm / (1.0 - rho) ** 2


@torch.no_grad()
def verify_contractivity(
    F: Callable[[Tensor], Tensor],
    z_star: Tensor,
    method: str = "auto",
    n_iter: int = 50,
) -> dict:
    """Full contractivity verification report.

    Returns:
        dict with keys: rho, is_contractive, margin (1-rho), method_used
    """
    rho = spectral_radius(F, z_star, method=method, n_iter=n_iter)
    return {
        "rho": rho,
        "is_contractive": rho < 1.0,
        "margin": max(0.0, 1.0 - rho),
        "method_used": method,
    }
