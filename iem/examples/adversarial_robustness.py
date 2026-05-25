"""Adversarial Equilibrium Theory — full empirical validation.

Demonstrates all three theorems + proposition on IGNN/Cora:

  1. Bound tightness: predicted sigma_1(S)*eps vs actual ||Dz*||
  2. Attack effectiveness: optimal IFT attack vs random perturbation
  3. Phase transition: vulnerability diverges as rho -> 1
  4. Per-node certificates: deterministic robust radii
  5. Non-normality index: spectral radius insufficiency

Usage:
    .venv/bin/python -m iem.examples.adversarial_robustness
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    certified_shift_bound,
    critical_perturbation_budget,
    extract_W_spectral_norm,
    full_adversarial_analysis,
    nonnormality_index,
    optimal_structural_attack,
    per_node_robust_radius,
    phase_transition_scan,
    structural_sensitivity_matrix,
    validate_bound_tightness,
    _compute_structural_jacobian,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/cora")

    # ---- Load + Train ----
    print("=== Loading Cora + Training IGNN ===", flush=True)
    _download_cora(data_dir)
    data = _load_cora(data_dir)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    import torch.nn.functional as F_func

    for ep in range(1, 101):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A_hat)
                pred = logits.argmax(dim=1)
                acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
            print(f"  ep {ep} | loss {loss.item():.4f} | test_acc {acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    print(f"  Final test_acc: {test_acc:.3f}\n")

    # ---- 50-node subgraph for tractable analysis ----
    deg = A_hat.sum(dim=1)
    center = int(deg.argmax().item())
    neighbors = (A_hat[center] > 0).nonzero(as_tuple=True)[0]
    idx = neighbors[:50]
    S_size = len(idx)

    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    Z_sub = Z_star[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}
    labels_sub = y[idx]

    # Reconverge to true subgraph fixed point (Z_sub from full graph is NOT
    # a fixed point of the subgraph operator due to missing cross-edges)
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new
    logits_sub = model.head(Z_sub)

    n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
    print(f"=== Subgraph: {S_size} nodes, {n_edges} edges around node {center} ===\n", flush=True)

    def F_sub(z, c):
        return model.operator(z, c)

    # ---- Spectral radius ----
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)
    print(f"Spectral radius rho = {rho:.4f} ({'contractive' if rho < 1 else 'NOT contractive'})\n")

    # ==================================================================
    # Theorem 1: Certified Fixed-Point Shift Bound
    # ==================================================================
    print("=" * 60, flush=True)
    print("THEOREM 1: Certified Fixed-Point Shift Bound", flush=True)
    print("=" * 60, flush=True)

    J_z, J_A, _ = _compute_structural_jacobian(F_sub, Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(F_sub, Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    print(f"  S shape: {S.shape}")

    eps = 0.01
    bound = certified_shift_bound(S, rho, eps, A_hat=A_sub)
    print(f"  sigma_1(S) unconstrained = {bound['sigma_1']:.4f}")
    if "constrained_sigma_1" in bound:
        print(f"  sigma_1(S_c) constrained  = {bound['constrained_sigma_1']:.4f} ({bound['n_edges']} edges)")
    print(f"  Unconstrained bound: {bound['upper_bound']:.6f}")
    if "constrained_upper_bound" in bound:
        print(f"  Constrained bound:   {bound['constrained_upper_bound']:.6f}")

    # Empirical validation
    print("\n  --- Bound tightness (constrained: symmetric, edge-only) ---")
    tightness = validate_bound_tightness(F_sub, model, Z_sub, ctx_sub, S, epsilons=[0.001, 0.005, 0.01, 0.05])
    print(f"  {'eps':>8} {'pred_constr':>11} {'actual':>10} {'random':>10} {'constr_tight':>12} {'unconstr_tight':>14} {'atk_adv':>8}")
    for r in tightness:
        print(f"  {r['epsilon']:>8.3f} {r['predicted_constr']:>11.6f} {r['actual_constr']:>10.6f} "
              f"{r['actual_random']:>10.6f} {r['constr_tightness']:>12.3f} {r['unconstr_tightness']:>14.3f} {r['attack_advantage']:>7.2f}x")

    # ==================================================================
    # Proposition 1: Optimal Structural Attack
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print("PROPOSITION 1: Optimal First-Order Structural Attack", flush=True)
    print("=" * 60, flush=True)

    attack = optimal_structural_attack(S, A_sub, epsilon=eps)
    print(f"  Max first-order shift: {attack['max_first_order_shift']:.6f}")
    print(f"  Effective adversarial dimensionality: {attack['effective_adversarial_dim']}")
    print(f"\n  Top-5 most vulnerable edges:")
    for i, j, v in attack['vulnerability_spectrum']:
        real_i, real_j = int(idx[i].item()), int(idx[j].item())
        print(f"    edge ({real_i}, {real_j}): vulnerability = {v:.4f}")

    # ==================================================================
    # Theorem 1(b,c): Critical Perturbation Budget
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print("THEOREM 1(b,c): Critical Perturbation Budget", flush=True)
    print("=" * 60, flush=True)

    W_norm = extract_W_spectral_norm(model)
    budget = critical_perturbation_budget(rho, W_norm)
    print(f"  ||W||_2 = {W_norm:.4f}")
    print(f"  Contractivity margin: 1 - rho = {budget['margin']:.4f}")
    print(f"  eps_crit >= {budget['epsilon_crit']:.4f}")
    print(f"  Interpretation: perturbations with ||dA||_F < {budget['epsilon_crit']:.4f} preserve contractivity")

    # Phase transition scan
    print("\n  --- Phase transition scan ---")
    pt = phase_transition_scan(model, Z_sub, ctx_sub, rho_targets=[0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99])
    print(f"  {'rho_target':>10} {'rho_actual':>10} {'actual_shift':>12} {'1/(1-rho)*eps':>14} {'converged':>9}")
    for r in pt:
        shift_s = f"{r['actual_shift']:>12.6f}" if r['actual_shift'] < 1e6 else "       DIVERGE"
        print(f"  {r['rho_target']:>10.2f} {r['rho_actual']:>10.4f} {shift_s} "
              f"{r['predicted_1_over_1mrho']:>14.6f} {str(r['converged']):>9}")

    # ==================================================================
    # Proposition 2: Per-Node Robust Radius
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print("PROPOSITION 2: Per-Node Robust Radius", flush=True)
    print("=" * 60, flush=True)

    node_certs = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    print(f"  Mean radius:   {node_certs['mean_radius']:.6f}")
    print(f"  Median radius: {node_certs['median_radius']:.6f}")
    print(f"  Fraction with non-trivial certificate: {node_certs['frac_nontrivial']:.1%}")
    print(f"  Fraction correctly classified + certified: {node_certs['frac_correct_and_certified']:.1%}")

    radii = node_certs["radii"]
    nontrivial = radii[radii > 1e-6]
    if len(nontrivial) > 0:
        print(f"  Non-trivial radii: min={float(nontrivial.min()):.4f}, "
              f"max={float(nontrivial.max()):.4f}, median={float(nontrivial.median()):.4f}")

    # ==================================================================
    # Non-normality analysis (Remark)
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print("REMARK: Non-Normality Index", flush=True)
    print("=" * 60, flush=True)

    nn = nonnormality_index(J_z, rho)
    print(f"  ||(I-J)^{{-1}}||_2 = {nn['resolvent_norm']:.4f}")
    print(f"  1/(1-rho) = {nn['naive_bound']:.4f}")
    print(f"  Non-normality index eta = {nn['nonnormality_index']:.4f}")
    print(f"  {nn['interpretation']}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print("ADVERSARIAL EQUILIBRIUM THEORY — SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  rho = {rho:.4f}")
    print(f"  sigma_1(S) = {bound['sigma_1']:.4f} (tight adversarial constant)")
    print(f"  eps_crit = {budget['epsilon_crit']:.4f} (contractivity threshold)")
    print(f"  eta = {nn['nonnormality_index']:.4f} (non-normality amplification)")
    if tightness:
        r = tightness[2] if len(tightness) > 2 else tightness[-1]
        print(f"  Constrained tightness at eps={r['epsilon']}: {r['constr_tightness']:.3f}")
        print(f"  Unconstrained tightness: {r['unconstr_tightness']:.3f}")
        print(f"  Optimal attack {r['attack_advantage']:.1f}x more effective than random")
    if node_certs:
        print(f"  {node_certs['frac_correct_and_certified']:.0%} nodes certifiably robust")
    print()


if __name__ == "__main__":
    sys.exit(main() or 0)
