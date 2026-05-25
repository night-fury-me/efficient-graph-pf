"""Node attribution for DEQ equilibrium models.

Three modes, ordered by speed vs. axiomatic fidelity:

  1. ift_attribution: O(n) gradient-based attribution from IFT sensitivities.
     Fast, but does NOT satisfy Shapley axioms (efficiency, symmetry, etc.).
     Use as a practical proxy; validate against exact_shapley on small graphs.

  2. exact_shapley: O(n · 2^n) coalition enumeration. True Shapley values
     satisfying all four axioms. Feasible for n ≤ 20.

  3. sampling_shapley: O(n · K) antithetic permutation sampling. Unbiased
     Shapley estimator for any n. Converges to exact with K → ∞.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def exact_shapley(
    value_fn: Callable[[Tensor], Tensor],
    z_star: Tensor,
    baseline: Tensor,
    player_dim: int,
    n_players: int | None = None,
) -> Tensor:
    """Exact Shapley values via coalition enumeration.

    Args:
        value_fn: maps state z (same shape as z_star) -> scalar value
        z_star: full coalition (all players present)
        baseline: empty coalition (no players / default state)
        player_dim: dimension along which players are indexed
        n_players: number of players (inferred from z_star if None)

    Returns:
        phi: (n_players,) Shapley values satisfying Σφ_i = V(z*) - V(baseline)
    """
    if n_players is None:
        n_players = z_star.shape[player_dim]

    if n_players > 20:
        raise ValueError(
            f"Exact Shapley with n={n_players} requires 2^{n_players} "
            f"evaluations. Use sampling_shapley for n > 20."
        )

    phi = torch.zeros(n_players, device=z_star.device, dtype=torch.float32)

    for i in range(n_players):
        for S_mask in range(1 << n_players):
            if S_mask & (1 << i):
                continue
            S_size = bin(S_mask).count("1")
            coeff = (
                math.factorial(S_size)
                * math.factorial(n_players - S_size - 1)
                / math.factorial(n_players)
            )
            z_S = _apply_coalition(baseline, z_star, S_mask, player_dim)
            z_Si = _apply_coalition(baseline, z_star, S_mask | (1 << i), player_dim)
            marginal = value_fn(z_Si) - value_fn(z_S)
            phi[i] += coeff * float(marginal.item())

    return phi


def sampling_shapley(
    value_fn: Callable[[Tensor], Tensor],
    z_star: Tensor,
    baseline: Tensor,
    player_dim: int,
    n_players: int | None = None,
    n_samples: int = 200,
    antithetic: bool = True,
    seed: int = 42,
) -> Tensor:
    """Approximate Shapley values via permutation sampling.

    Uses antithetic sampling (pair each permutation π with its reverse)
    for variance reduction.

    Args:
        value_fn: maps state z -> scalar value
        z_star: full coalition state
        baseline: empty coalition state
        player_dim: dimension along which players are indexed
        n_players: number of players
        n_samples: number of permutations to sample
        antithetic: if True, pair each π with reversed(π)
        seed: random seed

    Returns:
        phi: (n_players,) approximate Shapley values
    """
    if n_players is None:
        n_players = z_star.shape[player_dim]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.zeros(n_players, device=z_star.device, dtype=torch.float32)
    n_perms = 0

    for _ in range(n_samples):
        perm = torch.randperm(n_players, generator=gen)
        perms = [perm]
        if antithetic:
            perms.append(perm.flip(0))

        for pi in perms:
            mask = 0
            prev_val = value_fn(_apply_coalition(baseline, z_star, mask, player_dim))
            for idx in pi.tolist():
                mask = mask | (1 << idx)
                cur_val = value_fn(_apply_coalition(baseline, z_star, mask, player_dim))
                phi[idx] += float((cur_val - prev_val).item())
                prev_val = cur_val
            n_perms += 1

    phi /= n_perms
    return phi


def ift_attribution(
    node_sensitivity: Tensor,
    output_weights: Tensor | None = None,
) -> Tensor:
    """Fast O(n) node attribution from IFT sensitivities.

    Computes per-node importance as ||∂z*/∂x_i|| (L2 norm of the IFT
    sensitivity vector). This is a GRADIENT-BASED ATTRIBUTION, not a
    Shapley value — it does not satisfy the efficiency axiom in general.

    For true Shapley values, use exact_shapley() or sampling_shapley()
    which perform coalition enumeration / permutation sampling.

    Args:
        node_sensitivity: per-node sensitivity norms from param_sensitivity.
            Shape may be (N,), (1, N), or (N, D). All are flattened to (N,).
        output_weights: optional (D,) weights. If None, uses L2 norm.

    Returns:
        phi: (N,) per-node attribution scores (gradient-based, NOT Shapley)
    """
    # param_sensitivity returns shape matching param.shape, e.g. (1, N).
    # Squeeze batch dims so we always work with (N,) or (N, D).
    node_sensitivity = node_sensitivity.squeeze()

    if node_sensitivity.dim() <= 1:
        return node_sensitivity.abs()

    if output_weights is not None:
        return (node_sensitivity * output_weights.unsqueeze(0)).sum(-1).abs()

    return node_sensitivity.norm(dim=-1)


def _apply_coalition(
    baseline: Tensor,
    full: Tensor,
    mask: int,
    player_dim: int,
) -> Tensor:
    """Build coalition state: player i present if bit i is set in mask."""
    result = baseline.clone()
    n = full.shape[player_dim]
    for i in range(n):
        if mask & (1 << i):
            idx = [slice(None)] * full.dim()
            idx[player_dim] = i
            result[tuple(idx)] = full[tuple(idx)]
    return result
