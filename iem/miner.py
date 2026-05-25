"""High-level IEM API: mine equilibrium models in one call.

Usage:
    from iem import IEMiner

    # After running a DEQ model to convergence:
    miner = IEMiner(
        operator_fn=model._operator,    # F: (z, ctx) -> z'
        z_star=z_star,                  # converged fixed point
        ctx=ctx,                        # graph structure + features
    )

    # 1. Edge criticality (N-1 contingency mining)
    edge_sens = miner.edge_sensitivity("Y")          # ∂z*/∂Y_ij
    ranking = edge_sens.argsort(descending=True)      # most critical first

    # 2. Node attribution (IFT gradient-based, NOT Shapley)
    node_attr = miner.node_attribution("P_set")

    # 3. Certification
    report = miner.certify()
    print(f"Contractive: {report['is_contractive']}, ρ={report['rho']:.4f}")
    bound = miner.certified_bound(edge_sens)

Domain-agnostic: works with PE_DEQ_PF (power flow), IGNN (citation),
or any model exposing operator(z, ctx) -> z'.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from . import adversarial as _adv
from . import certify as _certify
from . import ift as _ift
from . import shapley as _shapley


class IEMiner:
    """Implicit Equilibrium Miner — domain-agnostic mining on DEQ fixed points.

    Args:
        operator_fn: the fixed-point operator F(z, ctx) -> z'
        z_star: the converged equilibrium state
        ctx: context dict (graph structure, features, masks, etc.)
        method: IFT solver method ("auto", "direct", "neumann")
    """

    def __init__(
        self,
        operator_fn: Callable[[Tensor, dict], Tensor],
        z_star: Tensor,
        ctx: dict,
        method: str = "auto",
    ):
        self.F = operator_fn
        self.z_star = z_star.detach()
        self.ctx = ctx
        self.method = method
        self._rho: float | None = None

    def _F_z_only(self, z: Tensor) -> Tensor:
        """Wrap F(z, ctx) to F(z) for Jacobian computation."""
        return self.F(z.reshape(self.z_star.shape), self.ctx).reshape(-1)

    # ----- Edge-level mining ------------------------------------------------

    def edge_sensitivity(
        self,
        edge_param_key: str,
        aggregate: str = "l2",
    ) -> Tensor:
        """Compute per-edge sensitivity ||∂z*/∂w_ij|| via IFT.

        Args:
            edge_param_key: key in ctx for edge parameters (e.g., "Y")
            aggregate: "l2" (norm per edge), "max" (max component), "raw" (full)

        Returns:
            sensitivity: per-edge scores (sorted by edge index)
        """
        raw = _ift.edge_sensitivity(
            self.F, self.z_star, self.ctx,
            edge_param_key, method=self.method,
        )
        if aggregate == "raw":
            return raw
        if aggregate == "l2":
            if raw.dim() > 1:
                return raw.reshape(raw.shape[0], -1).norm(dim=-1)
            return raw.abs()
        if aggregate == "max":
            if raw.dim() > 1:
                return raw.reshape(raw.shape[0], -1).abs().max(dim=-1).values
            return raw.abs()
        raise ValueError(f"Unknown aggregate: {aggregate}")

    def edge_ranking(self, edge_param_key: str) -> Tensor:
        """Rank edges by criticality (most critical first).

        Returns:
            indices: edge indices sorted by descending sensitivity
        """
        sens = self.edge_sensitivity(edge_param_key)
        return sens.argsort(descending=True)

    # ----- Node-level mining ------------------------------------------------

    def node_sensitivity(
        self,
        node_param_key: str,
        aggregate: str = "l2",
    ) -> Tensor:
        """Compute per-node sensitivity ||∂z*/∂x_i|| via IFT."""
        raw = _ift.node_sensitivity(
            self.F, self.z_star, self.ctx,
            node_param_key, method=self.method,
        )
        if aggregate == "raw":
            return raw
        if aggregate == "l2":
            if raw.dim() > 1:
                return raw.reshape(raw.shape[0], -1).norm(dim=-1)
            return raw.abs()
        if aggregate == "max":
            if raw.dim() > 1:
                return raw.reshape(raw.shape[0], -1).abs().max(dim=-1).values
            return raw.abs()
        raise ValueError(f"Unknown aggregate: {aggregate}")

    def node_attribution(
        self,
        node_param_key: str,
    ) -> Tensor:
        """Compute per-node IFT-based attribution (gradient importance).

        Returns ||∂z*/∂x_i|| for each node i — an O(n) gradient-based
        attribution. NOT Shapley values (does not satisfy efficiency axiom).
        For true Shapley, use exact_shapley() or sampling_shapley() from
        iem.shapley directly.

        Args:
            node_param_key: key in ctx for node features

        Returns:
            phi: (N,) per-node attribution scores
        """
        raw = _ift.node_sensitivity(
            self.F, self.z_star, self.ctx,
            node_param_key, method=self.method,
        )
        return _shapley.ift_attribution(raw)

    # Backward-compat alias (deprecated)
    node_shapley = node_attribution

    # ----- Certification ---------------------------------------------------

    @property
    def rho(self) -> float:
        """Spectral radius ρ(∂F/∂z) at z*. Cached after first computation."""
        if self._rho is None:
            self._rho = _certify.spectral_radius(
                self._F_z_only, self.z_star,
            )
        return self._rho

    def certify(self) -> dict:
        """Full contractivity verification report."""
        return _certify.verify_contractivity(self._F_z_only, self.z_star)

    # ----- Adversarial analysis (Theorems 1-3) ----------------------------

    def adversarial_analysis(
        self,
        model=None,
        A_key: str = "A_hat",
        epsilon: float = 0.01,
        logits: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict:
        """Full adversarial equilibrium analysis: Theorems 1-3 + Proposition 1.

        Returns certified shift bounds, optimal attack, critical perturbation
        budget, and per-node robust radii (if logits/labels provided).
        """
        return _adv.full_adversarial_analysis(
            self.F, model, self.z_star, self.ctx,
            A_key=A_key, epsilon=epsilon,
            logits=logits, labels=labels,
        )

    def structural_sensitivity(self, A_key: str = "A_hat") -> Tensor:
        """Compute the structural sensitivity matrix S = (I-J)^{-1} J_A."""
        return _adv.structural_sensitivity_matrix(
            self.F, self.z_star, self.ctx, A_key=A_key,
        )

    def certified_bound(
        self,
        sensitivity: Tensor,
        epsilon: float | None = None,
    ) -> dict:
        """Certified bound on sensitivity + optional ranking stability.

        Args:
            sensitivity: per-edge or per-node sensitivity magnitudes
            epsilon: optional adversarial budget for ranking stability

        Returns:
            dict with: rho, bound_per_element, max_bound, ranking_stability (if ε given)
        """
        rho = self.rho
        dF_dp_norms = sensitivity.detach()

        bounds = torch.tensor(
            [_certify.certified_sensitivity_bound(float(s), rho) for s in dF_dp_norms],
            device=sensitivity.device,
        )

        result = {
            "rho": rho,
            "is_contractive": rho < 1.0,
            "bound_per_element": bounds,
            "max_bound": float(bounds.max().item()),
        }

        if epsilon is not None:
            rank_stability = torch.tensor(
                [_certify.ranking_stability_bound(float(s), rho, epsilon) for s in dF_dp_norms],
                device=sensitivity.device,
            )
            result["ranking_stability"] = rank_stability
            result["max_rank_perturbation"] = float(rank_stability.max().item())
            result["epsilon"] = epsilon

        return result
