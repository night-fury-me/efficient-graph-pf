"""Mettack baseline: IFT vulnerability vs meta-gradient structural attack.

Implements Meta-Self variant of Mettack (Zügner et al., ICLR 2019) using a
2-layer GCN surrogate, then compares attack effectiveness against IFT
vulnerability ranking. Both evaluated by IGNN fixed-point shift.

Comparison at budget k = 1..5 edge flips:
  - IFT top-k: remove edges by IFT vulnerability score
  - Mettack top-k: remove edges by surrogate meta-gradient
  - Random: remove k random edges (averaged over 10 trials)
  - Greedy brute-force (k=1 only): ground truth

Usage:
    .venv/bin/python -m iem.examples.mettack_comparison
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    extract_ego_subgraph,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


# ---------------------------------------------------------------------------
# Surrogate GCN for Mettack
# ---------------------------------------------------------------------------

class SurrogateGCN(nn.Module):
    """2-layer GCN used as surrogate for meta-gradient computation."""

    def __init__(self, n_features: int, hidden: int, n_classes: int):
        super().__init__()
        self.gc1 = nn.Linear(n_features, hidden)
        self.gc2 = nn.Linear(hidden, n_classes)

    def forward(self, X, A_hat):
        H = F_func.relu(A_hat @ self.gc1(X))
        return A_hat @ self.gc2(H)


# ---------------------------------------------------------------------------
# Mettack (Meta-Self variant)
# ---------------------------------------------------------------------------

def mettack_edge_scores(
    X_sub: torch.Tensor,
    A_sub: torch.Tensor,
    pseudo_labels: torch.Tensor,
    n_features: int,
    n_classes: int,
    hidden: int = 64,
    train_epochs: int = 100,
) -> list:
    """Compute per-edge attack scores via surrogate GCN meta-gradient.

    Meta-Self variant: uses pseudo-labels from the target model, trains
    a surrogate GCN, then computes grad of attack loss w.r.t. adjacency.

    Returns list of (i, j, score) sorted by descending attack priority.
    """
    device = X_sub.device
    N = X_sub.shape[0]

    surrogate = SurrogateGCN(n_features, hidden, n_classes).to(device)
    optim = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)

    # Train surrogate on pseudo-labels
    for _ in range(train_epochs):
        surrogate.train()
        logits = surrogate(X_sub, A_sub)
        loss = F_func.cross_entropy(logits, pseudo_labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Compute meta-gradient: ∂L_attack / ∂A
    surrogate.eval()
    A_diff = A_sub.clone().detach().requires_grad_(True)
    logits = surrogate(X_sub, A_diff)
    # Attack loss: maximize misclassification = minimize correct-class logit
    loss_atk = -F_func.cross_entropy(logits, pseudo_labels)
    loss_atk.backward()

    grad = A_diff.grad.clone()
    grad.fill_diagonal_(0)
    # Symmetrize gradient
    grad = (grad + grad.T) / 2

    # Score each existing edge by |gradient|
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_sub[i, j].abs() > 1e-10:
                score = float(grad[i, j].abs())
                edges.append((i, j, score))

    edges.sort(key=lambda x: x[2], reverse=True)
    return edges


# ---------------------------------------------------------------------------
# Evaluate attack damage on IGNN
# ---------------------------------------------------------------------------

def evaluate_attack(
    model: nn.Module,
    z_clean: torch.Tensor,
    ctx: dict,
    edges_to_remove: list,
    A_key: str = "A_hat",
    reconverge_iter: int = 200,
) -> float:
    """Remove specified edges, reconverge IGNN, return ||z*_pert - z*_clean||."""
    A = ctx[A_key].clone()
    for i, j in edges_to_remove:
        A[i, j] = 0.0
        A[j, i] = 0.0
    ctx_pert = {**ctx, A_key: A}

    Z = z_clean.clone()
    with torch.no_grad():
        for _ in range(reconverge_iter):
            Z_new = model.operator(Z, ctx_pert)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return float((Z_new - z_clean).norm())


def run_comparison(name, data, device):
    """Full Mettack vs IFT comparison on one dataset."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}: N={data['N']}", flush=True)
    print(f"{'='*70}", flush=True)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    # Train IGNN
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(100):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    print(f"  IGNN test_acc={acc:.3f}", flush=True)

    # 50-node subgraph via BFS (guarantees connectivity)
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    S_size = len(idx)

    A_sub = A_hat[idx][:, idx]
    X_sub = X[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    labels_sub = y[idx]

    # Reconverge subgraph
    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    # Pseudo-labels for Mettack (Meta-Self: use IGNN predictions)
    with torch.no_grad():
        pseudo_labels = model.head(Z_sub).argmax(dim=1)

    n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
    print(f"  Subgraph: {S_size} nodes, {n_edges} edges", flush=True)

    if n_edges < 3:
        print(f"  SKIP: too few edges", flush=True)
        return None

    # --- IFT vulnerability ranking ---
    t0 = time.time()
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    ift_edges = [(i, j) for i, j, _ in attack["all_edge_vulnerabilities"]]
    ift_time = time.time() - t0
    print(f"  IFT: {len(ift_edges)} edges ranked in {ift_time:.1f}s", flush=True)

    # --- Mettack ranking ---
    t0 = time.time()
    mettack_ranked = mettack_edge_scores(
        X_sub, A_sub, pseudo_labels,
        n_features=data["n_features"], n_classes=data["n_classes"],
    )
    mettack_edges = [(i, j) for i, j, _ in mettack_ranked]
    mettack_time = time.time() - t0
    print(f"  Mettack: {len(mettack_edges)} edges ranked in {mettack_time:.1f}s", flush=True)

    # --- Greedy brute-force ranking ---
    t0 = time.time()
    bf = greedy_structural_attack(model, Z_sub, ctx_sub)
    bf_edges = [(i, j) for i, j, _ in bf]
    bf_time = time.time() - t0
    print(f"  Brute-force: {len(bf_edges)} edges ranked in {bf_time:.1f}s", flush=True)

    # --- Kendall tau: each method vs brute-force ground truth ---
    bf_rank = {(i, j): r for r, (i, j, _) in enumerate(bf)}

    def tau_vs_bf(method_edges, method_name):
        common = []
        for rank, (i, j) in enumerate(method_edges):
            key = (min(i, j), max(i, j))
            key_alt = (i, j)
            bf_r = bf_rank.get(key, bf_rank.get(key_alt, bf_rank.get((j, i), None)))
            if bf_r is not None:
                common.append((rank, bf_r))
        if len(common) < 3:
            return None
        a, b = zip(*common)
        tau, p = kendalltau(a, b)
        return tau

    tau_ift = tau_vs_bf(ift_edges, "IFT")
    tau_mettack = tau_vs_bf(mettack_edges, "Mettack")

    print(f"\n  Ranking correlation vs brute-force (ground truth):", flush=True)
    print(f"    IFT:     tau = {tau_ift:+.3f}" if tau_ift is not None else "    IFT:     N/A", flush=True)
    print(f"    Mettack: tau = {tau_mettack:+.3f}" if tau_mettack is not None else "    Mettack: N/A", flush=True)

    # --- Damage comparison at budget k = 1..min(5, n_edges) ---
    max_k = min(5, n_edges)
    print(f"\n  Damage (IGNN ||Δz*||) at budget k=1..{max_k}:", flush=True)
    print(f"  {'k':>3} {'IFT':>10} {'Mettack':>10} {'Random':>10} {'BruteForce':>10}", flush=True)

    results_by_k = []
    for k in range(1, max_k + 1):
        # IFT: remove top-k by vulnerability
        ift_remove = ift_edges[:k]
        dmg_ift = evaluate_attack(model, Z_sub, ctx_sub, ift_remove)

        # Mettack: remove top-k by meta-gradient
        met_remove = mettack_edges[:k]
        dmg_met = evaluate_attack(model, Z_sub, ctx_sub, met_remove)

        # Random: average over 10 trials
        import random
        all_edges = [(i, j) for i in range(S_size) for j in range(i+1, S_size) if A_sub[i,j].abs() > 1e-10]
        dmg_rands = []
        for _ in range(10):
            rand_remove = random.sample(all_edges, min(k, len(all_edges)))
            dmg_rands.append(evaluate_attack(model, Z_sub, ctx_sub, rand_remove))
        dmg_rand = sum(dmg_rands) / len(dmg_rands)

        # Brute-force: remove top-k by actual damage
        bf_remove = bf_edges[:k]
        dmg_bf = evaluate_attack(model, Z_sub, ctx_sub, bf_remove)

        print(f"  {k:>3} {dmg_ift:>10.4f} {dmg_met:>10.4f} {dmg_rand:>10.4f} {dmg_bf:>10.4f}", flush=True)
        results_by_k.append({
            "k": k, "ift": dmg_ift, "mettack": dmg_met,
            "random": dmg_rand, "bruteforce": dmg_bf,
        })

    return {
        "name": name,
        "n_edges": n_edges,
        "tau_ift": tau_ift,
        "tau_mettack": tau_mettack,
        "ift_time": ift_time,
        "mettack_time": mettack_time,
        "bf_time": bf_time,
        "damage_by_k": results_by_k,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    all_results = []
    for name, data in datasets:
        r = run_comparison(name, data, device)
        if r:
            all_results.append(r)

    # --- Summary ---
    print(f"\n\n{'='*70}", flush=True)
    print("METTACK vs IFT — SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Ranking correlation vs brute-force ground truth:", flush=True)
    print(f"  {'Dataset':<15} {'IFT tau':>10} {'Mettack tau':>12} {'IFT time':>10} {'Met time':>10} {'BF time':>10}", flush=True)
    for r in all_results:
        tau_i = f"{r['tau_ift']:+.3f}" if r['tau_ift'] is not None else "N/A"
        tau_m = f"{r['tau_mettack']:+.3f}" if r['tau_mettack'] is not None else "N/A"
        print(f"  {r['name']:<15} {tau_i:>10} {tau_m:>12} {r['ift_time']:>9.1f}s {r['mettack_time']:>9.1f}s {r['bf_time']:>9.1f}s", flush=True)

    print(f"\nDamage at k=1 (single edge removal):", flush=True)
    print(f"  {'Dataset':<15} {'IFT':>8} {'Mettack':>8} {'Random':>8} {'BF (best)':>10}", flush=True)
    for r in all_results:
        if r["damage_by_k"]:
            d = r["damage_by_k"][0]
            print(f"  {r['name']:<15} {d['ift']:>8.4f} {d['mettack']:>8.4f} {d['random']:>8.4f} {d['bruteforce']:>10.4f}", flush=True)

    # Winner analysis
    print(f"\nMethod comparison (higher damage = better attack):", flush=True)
    for r in all_results:
        ift_wins = sum(1 for d in r["damage_by_k"] if d["ift"] >= d["mettack"])
        met_wins = sum(1 for d in r["damage_by_k"] if d["mettack"] > d["ift"])
        total = len(r["damage_by_k"])
        print(f"  {r['name']}: IFT wins {ift_wins}/{total}, Mettack wins {met_wins}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
