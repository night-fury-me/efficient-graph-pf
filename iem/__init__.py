"""Implicit Equilibrium Mining (IEM) — domain-agnostic framework.

Given ANY differentiable fixed-point operator F_θ on a graph with
equilibrium z* = F_θ(z*), IEM extracts:

  1. Node/edge sensitivities via the implicit function theorem (IFT)
  2. Exact Shapley attribution from IFT gradients
  3. Certified sensitivity bounds from DEQ contractivity

The framework is domain-agnostic: theorems and algorithms depend only
on the contractive fixed-point structure, not on what F represents.
Demonstrated on: AC power flow, citation networks, traffic prediction.

Usage:
    from iem import IEMiner
    miner = IEMiner(operator_fn=model._operator, z_star=z_star, ctx=ctx)
    edge_sens = miner.edge_sensitivity(edge_params)
    node_shap = miner.node_shapley(node_params, n_samples=100)
    rho, bound = miner.certify(edge_sens)
"""

from .miner import IEMiner

__all__ = ["IEMiner"]
