"""Matrix-free AEGIS pipeline for scalable vulnerability analysis.

Replaces the dense O(D^3) pipeline in adversarial.py with:
  - Neumann series for (I - J_z)^{-1}: O(K*D) per solve
  - Autograd JVPs/VJPs for J_A: O(D) per operation
  - Randomized SVD for top-k: O(k * n_iter * K * D)

Dense pipeline: N <= 300 (24 GB at N=400).
Matrix-free:   N <= 5000+ (linear in D per operation).

Usage:
    op = ScalableSensitivity(F, z_star, ctx)
    U, sigma, Vh = op.top_k_svd(k=6)
    vulns = op.edge_vulnerability()
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ScalableSensitivity:
    """Matrix-free constrained sensitivity operator S_c.

    S_c maps edge perturbations to equilibrium shifts:
        S_c * v = (I - J_z)^{-1} * J_A * P_c * v

    where P_c projects edge vectors to symmetric adjacency perturbations.
    No large matrix is ever formed; all operations use JVPs/VJPs.
    """

    def __init__(
        self,
        F: Callable[[Tensor, dict], Tensor],
        z_star: Tensor,
        ctx: dict,
        A_key: str = "A_hat",
        neumann_terms: int = 0,
        neumann_tol: float = 1e-6,
        ignn_weight: Optional[Tensor] = None,
    ):
        """
        Args:
            neumann_terms: max Neumann iterations. 0 (default) = auto-detect
                from spectral radius so the series converges to neumann_tol.
            neumann_tol: early-stop when ||J_z^k b|| < tol * ||b||.
            ignn_weight: opt-in ANALYTIC path for an IGNN operator of the form
                ``F(Z) = ReLU(A_hat @ (Z @ W^T) + X_proj)``. When supplied with
                the *effective* (d, d) weight ``W`` (spectral scaling folded in,
                i.e. ``model._W_eff(Z) == Z @ W.T``), the four Jacobian
                applications use closed-form matmuls instead of autograd
                JVP/VJP. This removes the N x N backward graph that OOM'd
                full-graph Pubmed and is faster at any N. ``None`` (default)
                keeps the generic autograd path. Use ``extract_ignn_weight``
                to obtain ``W`` from a trained ``iem.examples.ignn_cora.IGNN``.
        """
        self.F = F
        self.z_star = z_star.detach()
        self.ctx = ctx
        self.A_key = A_key
        self.A = ctx[A_key].detach()
        self.N = self.A.shape[0]
        self.d = z_star.shape[-1] if z_star.dim() > 1 else z_star.numel() // self.N
        self.D = z_star.numel()
        self.neumann_tol = neumann_tol
        self.device = z_star.device
        self.dtype = z_star.dtype

        # ---- opt-in IGNN analytic Jacobian path (no autograd, no backward graph) ----
        # When `ignn_weight` is given, J_z / J_A and their transposes are applied
        # in closed form (all matmuls). Derived for F(Z) = ReLU(A_hat Z W^T + X_proj):
        #   J_z  v  = phi' ⊙ (A_hat V W^T)        J_z^T u  = A_hat (phi' ⊙ U) W
        #   J_A  δA = phi' ⊙ (δA Z W^T)           J_A^T u  = (phi' ⊙ U) W Z^T
        # with V/U = reshape to (N, d), Z = z_star, phi' = 1[z_star > 0] (ReLU mask
        # at the equilibrium, since z_star = ReLU(preact) ⇒ z_star>0 ⇔ preact>0),
        # A_hat symmetric, row-major (reshape(-1)) vec convention.
        self._analytic = ignn_weight is not None
        if self._analytic:
            if self.z_star.dim() != 2:
                raise ValueError(
                    "analytic IGNN path requires a 2-D (N, d) z_star; got shape "
                    f"{tuple(self.z_star.shape)}"
                )
            W = ignn_weight.detach().to(device=self.device, dtype=self.dtype)
            if W.shape != (self.d, self.d):
                raise ValueError(
                    f"ignn_weight must be (d, d)=({self.d}, {self.d}); got "
                    f"{tuple(W.shape)}"
                )
            self._W = W
            self._phi = (self.z_star > 0).to(self.dtype)  # (N, d) active mask
        else:
            self._W = None
            self._phi = None

        # Vectorized upper-triangular edge extraction. Identical to the nested
        # `for i: for j in range(i+1, N)` row-major scan (triu_indices yields the
        # same i<j order) but avoids ~N^2 GPU scalar syncs (156 s -> <1 s on Cora).
        iu = torch.triu_indices(self.N, self.N, offset=1, device=self.device)
        active = self.A[iu[0], iu[1]].abs() > 1e-10
        self._edge_idx = iu[:, active].t().contiguous().to(torch.long)
        self.edge_list: List[Tuple[int, int]] = [
            (int(i), int(j)) for i, j in self._edge_idx.tolist()
        ]
        self.num_edges = len(self.edge_list)
        if self.num_edges == 0:
            self._edge_idx = torch.zeros(0, 2, device=self.device, dtype=torch.long)

        self._f_base: Optional[Tensor] = None

        self.rho = self._estimate_rho()
        if neumann_terms > 0:
            self.neumann_K = neumann_terms
        else:
            self.neumann_K = self._adaptive_neumann_depth(self.rho, neumann_tol)

    @staticmethod
    def _adaptive_neumann_depth(rho: float, tol: float, cap: int = 3000) -> int:
        """K such that rho^K < tol, capped at `cap`.

        Cap raised to 3000 (was 500): at rho=0.99, K=500 leaves a ~e^{-5}~0.7%
        tail-term but the geometric resolvent error is ~rho^K/(1-rho) ~ 0.7%/0.01
        ~ 70% relative -> badly under-truncated. K=3000 drives rho^K/(1-rho) below
        the tol regime even at rho ~ 0.99 (Amazon Photo, rho ~ 1). Adaptive depth
        selection is unchanged; only the ceiling is raised.
        """
        if rho <= 0 or rho >= 1.0:
            return cap
        return min(cap, max(20, math.ceil(-math.log(tol) / math.log(1.0 / rho))))

    def _estimate_rho(self, n_iter: int = 150) -> float:
        """Spectral-radius estimate via power iteration + Rayleigh quotient on J_z.

        J_z is the fixed (linearised-at-z*) operator Jacobian. Power iteration
        converges to the dominant eigenvector v; the Rayleigh quotient <v, J_z v>
        is the dominant eigenvalue (sign-aware), so |<v, J_z v>| is the spectral
        radius rho. This replaces the previous estimate that returned the operator
        2-norm ||J_z v|| after only ~30 iters: ||J_z v|| upper-bounds rho and could
        OVERSHOOT a true rho ~ 0.96/0.99 to >= 1, which then silently pinned the
        Neumann depth at the cap (under-truncating). Mirrors `rho_rayleigh` in
        scripts/exp_fullgraph_attack_table.py.
        """
        v = torch.randn(self.D, device=self.device, dtype=self.dtype)
        v = v / v.norm()
        for _ in range(n_iter):
            Jv = self._jvp_Jz(v)
            nv = float(Jv.norm().item())
            if nv < 1e-12:
                return 0.0
            v = Jv / nv
        # Rayleigh quotient <v, J_z v> -> dominant eigenvalue (sign-aware).
        return abs(float((v * self._jvp_Jz(v)).sum().item()))

    # ------------------------------------------------------------------
    # Low-level JVP / VJP primitives
    # ------------------------------------------------------------------

    def _jvp_Jz(self, v: Tensor) -> Tensor:
        """J_z * v.  Analytic (IGNN): phi' ⊙ (A_hat V W^T); else forward-mode AD."""
        if self._analytic:
            V = v.reshape(self.N, self.d)
            return (self._phi * (self.A @ V @ self._W.t())).reshape(-1)
        z_flat = self.z_star.reshape(-1)
        try:
            from torch.func import jvp as torch_jvp

            def F_flat(z):
                return self.F(z.reshape(self.z_star.shape), self.ctx).reshape(-1)

            _, Jv = torch_jvp(F_flat, (z_flat,), (v,))
            return Jv
        except (ImportError, RuntimeError):
            eps = 1e-5
            with torch.no_grad():
                f_plus = self.F(
                    (z_flat + eps * v).reshape(self.z_star.shape), self.ctx
                ).reshape(-1)
                if self._f_base is None:
                    self._f_base = self.F(
                        z_flat.reshape(self.z_star.shape), self.ctx
                    ).reshape(-1)
                return (f_plus - self._f_base) / eps

    def _vjp_Jz(self, u: Tensor) -> Tensor:
        """J_z^T * u.  Analytic (IGNN): A_hat (phi' ⊙ U) W; else backward-mode AD."""
        if self._analytic:
            U = u.reshape(self.N, self.d)
            return (self.A @ (self._phi * U) @ self._W).reshape(-1)
        z = self.z_star.detach().reshape(-1).requires_grad_(True)
        F_val = self.F(z.reshape(self.z_star.shape), self.ctx).reshape(-1)
        (grad_z,) = torch.autograd.grad(
            F_val, z, grad_outputs=u, retain_graph=False, create_graph=False
        )
        return grad_z.detach()

    def _structural_jvp(self, delta_A: Tensor) -> Tensor:
        """J_A * vec(delta_A).  Analytic (IGNN): phi' ⊙ (δA Z W^T); else forward AD."""
        if self._analytic:
            return (self._phi * (delta_A @ self.z_star @ self._W.t())).reshape(-1)
        A_base = self.A
        try:
            from torch.func import jvp as torch_jvp

            def F_of_A(A_val):
                ctx_local = {**self.ctx, self.A_key: A_val}
                return self.F(self.z_star, ctx_local).reshape(-1)

            _, tangent = torch_jvp(F_of_A, (A_base,), (delta_A,))
            return tangent
        except (ImportError, RuntimeError):
            eps = 1e-4
            with torch.no_grad():
                ctx_pert = {**self.ctx, self.A_key: A_base + eps * delta_A}
                f_pert = self.F(self.z_star, ctx_pert).reshape(-1)
                if self._f_base is None:
                    self._f_base = self.F(self.z_star, self.ctx).reshape(-1)
                return (f_pert - self._f_base) / eps

    def _structural_vjp(self, u: Tensor) -> Tensor:
        """J_A^T * u. Returns (N, N) gradient w.r.t. A.

        Analytic (IGNN): (phi' ⊙ U) W Z^T; else backward-mode AD. The analytic
        form never builds the N x N backward graph that OOM'd full-graph Pubmed.
        """
        if self._analytic:
            U = u.reshape(self.N, self.d)
            return (self._phi * U) @ self._W @ self.z_star.t()
        A = self.A.detach().requires_grad_(True)
        ctx_grad = {**self.ctx, self.A_key: A}
        F_val = self.F(self.z_star.detach(), ctx_grad).reshape(-1)
        (grad_A,) = torch.autograd.grad(
            F_val, A, grad_outputs=u, retain_graph=False, create_graph=False
        )
        return grad_A.detach()

    # ------------------------------------------------------------------
    # Neumann series solvers
    # ------------------------------------------------------------------

    def _neumann_solve(self, rhs: Tensor) -> Tensor:
        """(I - J_z)^{-1} * rhs via truncated Neumann series."""
        result = rhs.clone()
        term = rhs.clone()
        b_norm = rhs.norm().item()
        if b_norm < 1e-15:
            return result
        for _ in range(self.neumann_K):
            term = self._jvp_Jz(term)
            result = result + term
            if term.norm().item() < self.neumann_tol * b_norm:
                break
        return result

    def _neumann_solve_adjoint(self, rhs: Tensor) -> Tensor:
        """(I - J_z^T)^{-1} * rhs via adjoint Neumann series."""
        result = rhs.clone()
        term = rhs.clone()
        b_norm = rhs.norm().item()
        if b_norm < 1e-15:
            return result
        for _ in range(self.neumann_K):
            term = self._vjp_Jz(term)
            result = result + term
            if term.norm().item() < self.neumann_tol * b_norm:
                break
        return result

    # ------------------------------------------------------------------
    # Edge ↔ adjacency mappings
    # ------------------------------------------------------------------

    def _edges_to_delta_A(self, v: Tensor) -> Tensor:
        """Map edge vector v in R^|E| to symmetric delta_A in R^{N x N}."""
        delta_A = torch.zeros(self.N, self.N, device=self.device, dtype=self.dtype)
        if self.num_edges == 0:
            return delta_A
        delta_A[self._edge_idx[:, 0], self._edge_idx[:, 1]] = v
        delta_A[self._edge_idx[:, 1], self._edge_idx[:, 0]] = v
        return delta_A

    def _delta_A_to_edges(self, delta_A: Tensor) -> Tensor:
        """Map delta_A in R^{N x N} to edge vector in R^|E|."""
        if self.num_edges == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        return (
            delta_A[self._edge_idx[:, 0], self._edge_idx[:, 1]]
            + delta_A[self._edge_idx[:, 1], self._edge_idx[:, 0]]
        )

    # ------------------------------------------------------------------
    # Matrix-free S_c operator
    # ------------------------------------------------------------------

    def matvec(self, v: Tensor) -> Tensor:
        """S_c * v: edge perturbation -> equilibrium shift."""
        delta_A = self._edges_to_delta_A(v)
        rhs = self._structural_jvp(delta_A)
        return self._neumann_solve(rhs)

    def rmatvec(self, u: Tensor) -> Tensor:
        """S_c^T * u: equilibrium vector -> edge sensitivities."""
        resolved = self._neumann_solve_adjoint(u)
        grad_A = self._structural_vjp(resolved)
        return self._delta_A_to_edges(grad_A)

    # ------------------------------------------------------------------
    # Randomized SVD (Halko, Martinsson, Tropp 2011)
    # ------------------------------------------------------------------

    def top_k_svd(
        self,
        k: int = 6,
        n_oversamples: int = 10,
        n_power_iter: int = 5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Top-k singular triplets via randomized SVD.

        Never forms S_c; uses matvec/rmatvec only.
        Cost: O((k + n_oversamples) * n_power_iter * K_neumann * D).

        Returns:
            U:     (D, k)    left singular vectors
            sigma: (k,)      singular values (descending)
            Vh:    (k, |E|)  right singular vectors
        """
        m, n = self.D, self.num_edges
        ell = min(k + n_oversamples, n)

        Omega = torch.randn(n, ell, device=self.device, dtype=self.dtype)
        Y = torch.zeros(m, ell, device=self.device, dtype=self.dtype)
        for j in range(ell):
            Y[:, j] = self.matvec(Omega[:, j])

        for _ in range(n_power_iter):
            Z = torch.zeros(n, ell, device=self.device, dtype=self.dtype)
            for j in range(ell):
                Z[:, j] = self.rmatvec(Y[:, j])
            Z, _ = torch.linalg.qr(Z)

            Y = torch.zeros(m, ell, device=self.device, dtype=self.dtype)
            for j in range(ell):
                Y[:, j] = self.matvec(Z[:, j])
            Y, _ = torch.linalg.qr(Y)

        Q, _ = torch.linalg.qr(Y)
        Q = Q[:, :ell]

        B = torch.zeros(ell, n, device=self.device, dtype=self.dtype)
        for i in range(ell):
            B[i, :] = self.rmatvec(Q[:, i])

        U_hat, sigma, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat

        return U[:, :k], sigma[:k], Vh[:k, :]

    # ------------------------------------------------------------------
    # Per-edge vulnerability
    # ------------------------------------------------------------------

    def _column(self, edge_idx: int) -> Tensor:
        """Compute S_c[:, edge_idx] without forming S_c."""
        i, j = self.edge_list[edge_idx]
        delta_A = torch.zeros(self.N, self.N, device=self.device, dtype=self.dtype)
        delta_A[i, j] = 1.0
        delta_A[j, i] = 1.0
        rhs = self._structural_jvp(delta_A)
        return self._neumann_solve(rhs)

    def edge_vulnerability(
        self,
        top_k: int = 0,
    ) -> List[Tuple[int, int, float]]:
        """Per-edge vulnerability ||S_c[:, k]||_2 via column computation.

        Args:
            top_k: return only top-k edges (0 = all, sorted descending)

        Returns:
            List of (i, j, vulnerability) sorted by descending vulnerability.
        """
        results: List[Tuple[int, int, float]] = []

        for idx in range(self.num_edges):
            col = self._column(idx)
            vuln = float(col.norm().item())
            i, j = self.edge_list[idx]
            results.append((i, j, vuln))

        results.sort(key=lambda x: x[2], reverse=True)
        if top_k > 0:
            results = results[:top_k]
        return results

    # ------------------------------------------------------------------
    # Per-node sensitivity norms (random probing)
    # ------------------------------------------------------------------

    def node_sensitivity_norms(self, n_probes: int = 20) -> Tensor:
        """Per-node ||S_v||_F via Hutchinson trace estimator.

        Uses random probes: E[||S_v g||^2] = ||S_v||_F^2 for g ~ N(0,I).

        Returns:
            (N,) per-node sensitivity Frobenius norms.
        """
        n_probes = min(n_probes, max(self.num_edges, 1))
        sq_norms = torch.zeros(self.N, device=self.device, dtype=self.dtype)

        for _ in range(n_probes):
            g = torch.randn(self.num_edges, device=self.device, dtype=self.dtype)
            Sg = self.matvec(g)
            Sg_mat = Sg.reshape(self.N, self.d)
            sq_norms += Sg_mat.pow(2).sum(dim=1)

        return (sq_norms / n_probes).sqrt()


# ----------------------------------------------------------------------
# IGNN effective-weight extraction (for the analytic Jacobian path)
# ----------------------------------------------------------------------


def extract_ignn_weight(model: nn.Module) -> Tensor:
    """Effective (d, d) weight ``W`` of a trained IGNN, spectral scaling folded in.

    Returns ``W`` such that ``model._W_eff(Z) == Z @ W.T`` exactly, so it can be
    passed to ``ScalableSensitivity(..., ignn_weight=W)`` to enable the closed-form
    Jacobian path. We probe rather than re-deriving the cap arithmetic: ``_W_eff``
    is linear in ``Z`` (the hard-cap ``scale`` is constant w.r.t. ``Z`` at a fixed,
    eval-mode model), so ``_W_eff(I_d) = I_d @ W.T = W.T``.

    Works for both the hard-cap (``c`` set) and legacy ``spectral_norm`` recipes,
    since both reduce ``_W_eff`` to a single linear map.
    """
    d = int(model.W.weight.shape[0])
    ref = model.W.weight
    eye = torch.eye(d, device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        W_eff_T = model._W_eff(eye)  # (d, d) = W_eff^T
    return W_eff_T.t().contiguous().detach()


# ----------------------------------------------------------------------
# High-level analysis functions
# ----------------------------------------------------------------------


def scalable_adversarial_analysis(
    F: Callable[[Tensor, dict], Tensor],
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    epsilon: float = 0.01,
    logits: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    k: int = 6,
    neumann_terms: int = 0,
    neumann_tol: float = 1e-6,
) -> dict:
    """Full AEGIS analysis via the matrix-free pipeline.

    Drop-in replacement for adversarial.full_adversarial_analysis
    that scales to N ~ 5000+.
    """
    from .adversarial import critical_perturbation_budget, extract_W_spectral_norm

    op = ScalableSensitivity(
        F, z_star, ctx, A_key,
        neumann_terms=neumann_terms,
        neumann_tol=neumann_tol,
    )
    rho = op.rho

    U, sigma, Vh = op.top_k_svd(k=k)
    sigma_1 = float(sigma[0])

    attack_weights = epsilon * Vh[0]
    attack_direction = op._edges_to_delta_A(attack_weights)
    attack_direction = (attack_direction + attack_direction.T) / 2
    norm = attack_direction.norm()
    if norm > 1e-10:
        attack_direction = attack_direction / norm * epsilon

    edge_vulns = op.edge_vulnerability()

    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = None
    budget = critical_perturbation_budget(rho, W_norm) if W_norm else None

    eff_dim = int((sigma > sigma[0] * 0.01).sum()) if sigma.numel() > 0 else 0

    node_certs = None
    if logits is not None and labels is not None and hasattr(model, "head"):
        node_certs = _scalable_node_radii(
            op, z_star, logits, labels, rho, model.head,
        )

    return {
        "rho": rho,
        "sigma_1": sigma_1,
        "sigma_spectrum": sigma.detach().cpu(),
        "max_first_order_shift": sigma_1 * epsilon,
        "attack_direction": attack_direction.detach(),
        "vulnerability_spectrum": edge_vulns[:5],
        "all_edge_vulnerabilities": edge_vulns,
        "effective_adversarial_dim": eff_dim,
        "epsilon_crit": budget,
        "node_certificates": node_certs,
        "method": "matrix_free",
        "n_nodes": op.N,
        "n_edges": op.num_edges,
    }


def _scalable_node_radii(
    op: ScalableSensitivity,
    z_star: Tensor,
    logits: Tensor,
    labels: Tensor,
    rho: float,
    head: nn.Module,
    n_probes: int = 20,
) -> dict:
    """Matrix-free per-node first-order radius for N > 200 -- runner-up product-norm SURROGATE.

    Uses one Hutchinson ||S_v||_F probe per node and the runner-up margin direction:
        r_hat_v = (f_{y_v} - f_{c*}) / (||W_{y_v} - W_{c*}||_2 ||S_v||_F),   y_v = argmax(logits)_v.
    This is the only affordable form at N ~ 7,650. The EXACT min-over-classes composed-norm radius
    of prop:radius (``per_node_robust_radius``, the dense-path default) needs N*C rmatvec passes
    (``||(W_{y_v}-W_c) S_v|| = ||rmatvec(e_v (x) (W_{y_v}-W_c))||``) and is intractable at scale;
    the surrogate under-estimates each per-class radius by Cauchy-Schwarz. No reported figure uses
    this path -- all reported r_v come from the exact dense path."""
    N = logits.shape[0]
    if rho >= 1.0:
        return {"radii": torch.zeros(N), "certified": False, "reason": "rho >= 1"}
    if not isinstance(head, nn.Linear):
        raise ValueError("matrix-free radius assumes a linear head (prop:radius)")

    W = head.weight.detach()
    L = logits.detach()
    preds = L.argmax(dim=1)
    S_node_norms = op.node_sensitivity_norms(n_probes=n_probes).to(L.device)

    radii = torch.zeros(N, device=L.device)
    margins = torch.zeros(N, device=L.device)
    for v in range(N):
        p = int(preds[v])
        marg = L[v, p] - L[v]
        marg[p] = float("inf")
        cstar = int(marg.argmin())
        margins[v] = marg[cstar]
        denom = (W[p] - W[cstar]).norm() * S_node_norms[v] + 1e-10
        radii[v] = (marg[cstar] / denom).clamp(min=0)

    correct = (preds.cpu() == labels.cpu()) if labels is not None \
        else torch.ones(N, dtype=torch.bool)
    return {
        "radii": radii.detach().cpu(),
        "margins": margins.detach().cpu(),
        "certified": True,
        "method": "runner_up_surrogate (matrix-free)",
        "mean_radius": float(radii.mean()),
        "median_radius": float(radii.median()),
        "frac_nontrivial": float((radii > 1e-6).float().mean()),
        "frac_correct_and_certified": float(
            (correct & (radii.cpu() > 1e-6)).float().mean()
        ),
    }


def auto_adversarial_analysis(
    F: Callable[[Tensor, dict], Tensor],
    model: nn.Module,
    z_star: Tensor,
    ctx: dict,
    A_key: str = "A_hat",
    epsilon: float = 0.01,
    logits: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    threshold_N: int = 200,
    **kwargs,
) -> dict:
    """Auto-select dense or matrix-free pipeline based on graph size.

    Uses the dense solver from adversarial.py when N <= threshold_N
    (exact, faster for small graphs). Switches to the matrix-free
    Neumann pipeline for larger graphs.

    Args:
        threshold_N: switch to matrix-free above this node count.
        **kwargs: forwarded to the chosen backend.
    """
    N = ctx[A_key].shape[0]

    if N <= threshold_N:
        from .adversarial import full_adversarial_analysis

        result = full_adversarial_analysis(
            F, model, z_star, ctx, A_key, epsilon, logits, labels,
        )
        result["method"] = "dense"
        return result

    return scalable_adversarial_analysis(
        F, model, z_star, ctx, A_key, epsilon, logits, labels, **kwargs,
    )
