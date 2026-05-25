"""Implicit Equilibrium Mining (IEM) — domain-agnostic framework.

Given ANY differentiable fixed-point operator F_θ on a graph with
equilibrium z* = F_θ(z*), IEM extracts:

  1. Node/edge sensitivities via the implicit function theorem (IFT)
  2. Gradient-based node attribution (O(n), fast proxy for importance)
  3. Certified sensitivity bounds from DEQ contractivity (ρ < 1)

The framework is domain-agnostic: all computations depend only on
the contractive fixed-point structure, not on what F represents.
Demonstrated on: AC power flow, citation networks, e-commerce, encyclopedia.

Usage:
    from iem import IEMiner
    miner = IEMiner(operator_fn=model._operator, z_star=z_star, ctx=ctx)
    edge_sens = miner.edge_sensitivity(edge_params)
    node_attr = miner.node_attribution(node_params)
    report = miner.certify()
"""

from .miner import IEMiner
from . import adversarial

__all__ = ["IEMiner", "adversarial"]
