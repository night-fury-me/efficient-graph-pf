"""Deep-Equilibrium machinery: Anderson-accelerated forward solve and
implicit-function-theorem backward.

Standard pattern from Bai, Kolter, Koltun (NeurIPS 2019). Anderson
acceleration is a quasi-Newton scheme that turns linear fixed-point
iteration into super-linear convergence; the backward pass solves the
adjoint fixed-point equation (I - dF/dz)^T g = dL/dz_star via one linear
solve, giving O(1) memory in iteration depth.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.autograd as autograd
import torch.nn as nn


def anderson_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    m: int = 5,
    lam: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-4,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, list[float]]:
    """Anderson acceleration for x = f(x).

    Args:
        f: fixed-point map. Must return a tensor with the same shape as x0.
        x0: initial guess. Shape (B, ...).
        m: history window size.
        lam: Tikhonov regularization for the coefficient least-squares.
        max_iter: max iterations (must be >= 2).
        tol: relative residual tolerance for early exit.
        beta: mixing parameter (1.0 = unmixed Anderson).

    Returns:
        x*: approximate fixed point with same shape as x0.
        res: per-iteration relative residual history.
    """
    if max_iter < 2:
        raise ValueError("max_iter must be >= 2")
    bsz = x0.shape[0]
    N = x0[0].numel()
    dtype, device = x0.dtype, x0.device

    X = torch.zeros(bsz, m, N, dtype=dtype, device=device)
    F = torch.zeros(bsz, m, N, dtype=dtype, device=device)
    X[:, 0] = x0.reshape(bsz, -1)
    F[:, 0] = f(x0).reshape(bsz, -1)
    X[:, 1] = F[:, 0]
    F[:, 1] = f(X[:, 1].view_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=dtype, device=device)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    y = torch.zeros(bsz, m + 1, 1, dtype=dtype, device=device)
    y[:, 0] = 1.0

    res: list[float] = []
    k_last = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=dtype, device=device).unsqueeze(0)
        )
        try:
            alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]
        except RuntimeError:
            break
        idx = k % m
        X[:, idx] = (
            beta * torch.einsum("bn,bnd->bd", alpha, F[:, :n])
            + (1.0 - beta) * torch.einsum("bn,bnd->bd", alpha, X[:, :n])
        )
        F[:, idx] = f(X[:, idx].view_as(x0)).reshape(bsz, -1)
        r = (F[:, idx] - X[:, idx]).norm(dim=-1) / (F[:, idx].norm(dim=-1) + 1e-8)
        res.append(float(r.max().item()))
        k_last = idx
        if res[-1] < tol:
            break
    return X[:, k_last].view_as(x0), res


def naive_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, list[float]]:
    """Plain fixed-point iteration x_{k+1} = f(x_k). Baseline / fallback."""
    x = x0
    res: list[float] = []
    for _ in range(max_iter):
        x_new = f(x)
        r = (x_new - x).flatten(1).norm(dim=-1) / (x_new.flatten(1).norm(dim=-1) + 1e-8)
        res.append(float(r.max().item()))
        x = x_new
        if res[-1] < tol:
            break
    return x, res


class DEQFixedPoint(nn.Module):
    """Wrap an operator f(z, *args) into a module whose forward returns
    z_star such that f(z_star, *args) = z_star.

    Forward: no-grad Anderson solve, then one autograd-enabled f-call to
    splice gradients into the rest of the graph.

    Backward: a tensor hook on z_star solves the adjoint fixed-point
    equation via the same solver -- implementing implicit-function-theorem
    gradients with O(1) memory in iteration depth.
    """

    def __init__(
        self,
        f: Callable,
        solver: Callable = anderson_solver,
        solver_kwargs: dict | None = None,
        backward_solver: Callable | None = None,
        backward_kwargs: dict | None = None,
    ):
        super().__init__()
        self.f = f
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}
        self.backward_solver = backward_solver or solver
        self.backward_kwargs = backward_kwargs or self.solver_kwargs
        self.forward_res: list[float] = []
        self.backward_res: list[float] = []

    def forward(self, z0: torch.Tensor, *args) -> torch.Tensor:
        # 1) No-grad forward solve
        with torch.no_grad():
            z_star, self.forward_res = self.solver(
                lambda z: self.f(z, *args), z0, **self.solver_kwargs
            )

        # 2) One f-call with autograd -> attach z_star to the graph.
        z_star = self.f(z_star, *args)

        # 3) IFT backward via a tensor hook on z_star. The hook replaces
        #    dL/dz_star with (I - dF/dz)^{-T} dL/dz_star, computed by
        #    re-using the same fixed-point solver on the adjoint map.
        if z_star.requires_grad:
            z_in = z_star.clone().detach().requires_grad_(True)
            f_out = self.f(z_in, *args)

            def backward_hook(grad: torch.Tensor) -> torch.Tensor:
                g_solved, self.backward_res = self.backward_solver(
                    lambda y: autograd.grad(f_out, z_in, y, retain_graph=True)[0] + grad,
                    grad,
                    **self.backward_kwargs,
                )
                return g_solved

            z_star.register_hook(backward_hook)

        return z_star
