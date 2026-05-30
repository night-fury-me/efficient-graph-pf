"""AEGIS diagnostic-only demo -- released unconditionally.

Computes and prints AEGIS's first-order structural-vulnerability DIAGNOSTICS for a trained
graph neural network:
  * v_ij : per-edge vulnerability scores -- which edges the equilibrium is most sensitive to;
  * r_v  : per-node first-order sensitivity radii -- how large a structural perturbation each
           node tolerates (to first order) before its prediction may flip.

It calls ``iem.adversarial.diagnostic_analysis``, which by construction produces only scalar
scores and radii: it does NOT synthesise an attack direction. The attack-direction synthesis
(Proposition 1 / ``optimal_structural_attack``) is intentionally NOT imported in this file --
it is gated per the paper's coordinated-disclosure protocol, since the diagnostic scores alone
cannot directly reconstruct a perturbation.

Usage:
    python scripts/aegis_diagnose.py        # Cora, 50-node ego-subgraph (exact dense path)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import diagnostic_analysis, extract_ego_subgraph
from iem.examples.ignn_cora import IGNN, _load_cora


def train(model, X, A_hat, y, mask, epochs=150, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _, _ = model(X, A_hat)
        F_func.cross_entropy(logits[mask], y[mask]).backward()
        opt.step()
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    data = _load_cora(Path("datasets/cora"))
    X, A_hat, y = data["X"].to(device), data["A_hat"].to(device), data["y"].to(device)
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    train(model, X, A_hat, y, data["train_mask"].to(device))

    # exact dense path on a 50-node ego-subgraph (N <= 200)
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)
    ctx_sub = {"A_hat": A_hat[idx][:, idx], "X_proj": ctx["X_proj"][idx]}
    Z = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(300):
            Zn = model.operator(Z, ctx_sub)
            if (Zn - Z).norm() < 1e-8:
                break
            Z = Zn
        logits_sub = model.head(Zn)

    diag = diagnostic_analysis(
        lambda z, c: model.operator(z, c), model, Zn, ctx_sub,
        logits=logits_sub, labels=y[idx],
    )

    ec = diag["eps_crit"]
    ec_str = f"{ec:.4f}" if ec is not None else "n/a"
    print(f"AEGIS diagnostics  (Cora, {len(idx)}-node ego-subgraph, {len(diag['v_ij'])} edges)")
    print(f"  rho(J_z) = {diag['rho']:.4f}    sigma_1(S_c) = {diag['sigma_1']:.4f}    "
          f"eps_crit = {ec_str}")
    print("\n  Top-10 most vulnerable edges (largest v_ij):")
    for (i, j), v in sorted(diag["v_ij"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"    edge ({i:>2d}, {j:>2d})    v_ij = {v:.4f}")
    if diag["r_v"] is not None:
        rv = diag["r_v"]
        print(f"\n  Per-node first-order radii r_v: median = {float(rv.median()):.4f}, "
              f"min = {float(rv.min()):.4f}   (smaller r_v = more vulnerable node)")
    print(f"\n  [{diag['note']}]")


if __name__ == "__main__":
    main()
