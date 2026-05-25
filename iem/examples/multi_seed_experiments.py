"""Multi-seed experiments for publication-ready results (mean ± std).

Runs ALL paper experiments across 10 seeds on 5 graph benchmark datasets:
  - Constrained tightness (Theorem 1)
  - Attack advantage over random (Proposition 1)
  - IFT vs Mettack damage + tau (Proposition 1 baseline)
  - Deterministic vs smoothing certificates (Proposition 2 baseline)

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.multi_seed_experiments
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    greedy_structural_attack,
    optimal_structural_attack,
    per_node_robust_radius,
    randomized_smoothing_certificate,
    structural_sensitivity_matrix,
    validate_bound_tightness,
    critical_perturbation_budget,
    extract_W_spectral_norm,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_amazon import _load_amazon
    from iem.examples.ignn_wikics import _load_wikics

    return [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("Pubmed", _load_planetoid("pubmed", Path("datasets/pubmed"))),
        ("Amazon", _load_amazon(Path("datasets/amazon_photo"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]


def run_single_seed(name, data, seed, device, run_mettack=False, run_smoothing=False):
    """Run all experiments for one dataset + one seed. Returns metrics dict."""
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(100):
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
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())

    # Subgraph
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    S_size = len(idx)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    labels_sub = y[idx]

    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new
    logits_sub = model.head(Z_sub)

    n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    # Compute S
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    # Constrained tightness
    tight_results = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.01], n_random=3,
    )
    constr_tight = tight_results[0]["constr_tightness"]
    atk_adv = tight_results[0]["attack_advantage"]

    # Critical budget
    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = 1.0
    budget = critical_perturbation_budget(rho, W_norm)
    eps_crit = budget["epsilon_crit"]

    # Per-node certificates
    node_certs = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    det_coverage = node_certs["frac_correct_and_certified"]
    det_med_r = node_certs["median_radius"]

    result = {
        "test_acc": test_acc, "rho": rho, "n_edges": n_edges,
        "constr_tight": constr_tight, "atk_adv": atk_adv,
        "eps_crit": eps_crit, "det_coverage": det_coverage, "det_med_r": det_med_r,
    }

    # Mettack comparison
    if run_mettack and n_edges >= 3:
        from iem.examples.mettack_comparison import mettack_edge_scores

        # IFT vulnerability
        attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
        ift_edges = [(i, j) for i, j, _ in attack["all_edge_vulnerabilities"]]

        # Brute-force
        bf = greedy_structural_attack(model, Z_sub, ctx_sub)
        bf_edges = [(i, j) for i, j, _ in bf]
        bf_rank = {(i, j): r for r, (i, j, _) in enumerate(bf)}

        # Mettack
        pseudo_labels = model.head(Z_sub).argmax(dim=1)
        X_sub = X[idx]
        met_ranked = mettack_edge_scores(
            X_sub, A_sub, pseudo_labels,
            n_features=data["n_features"], n_classes=data["n_classes"],
        )
        met_edges = [(i, j) for i, j, _ in met_ranked]

        def tau_vs_bf(method_edges):
            common = []
            for rank, (i, j) in enumerate(method_edges):
                key = (min(i, j), max(i, j))
                bf_r = bf_rank.get(key, bf_rank.get((i, j), bf_rank.get((j, i), None)))
                if bf_r is not None:
                    common.append((rank, bf_r))
            if len(common) < 3:
                return None
            a, b = zip(*common)
            tau, _ = kendalltau(a, b)
            return tau

        result["ift_tau"] = tau_vs_bf(ift_edges)
        result["met_tau"] = tau_vs_bf(met_edges)

        # Damage at k=1
        from iem.examples.mettack_comparison import evaluate_attack
        result["ift_dmg_k1"] = evaluate_attack(model, Z_sub, ctx_sub, ift_edges[:1])
        result["met_dmg_k1"] = evaluate_attack(model, Z_sub, ctx_sub, met_edges[:1])

    # Smoothing comparison
    if run_smoothing:
        smooth = randomized_smoothing_certificate(
            model, Z_sub, ctx_sub, labels_sub, sigma=0.01, n_samples=100,
        )
        result["smooth_coverage"] = smooth["frac_certified"]
        smooth_nontrivial = smooth["radii"][smooth["radii"] > 1e-6]
        result["smooth_med_r"] = float(smooth_nontrivial.median()) if len(smooth_nontrivial) > 0 else 0.0

    return result


def agg(vals):
    """Mean ± std string."""
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.3f}±{s:.3f}"


def agg_pct(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m*100:.1f}±{s*100:.1f}%"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("Loading datasets...", flush=True)
    datasets = load_all_datasets()
    print(f"  Loaded {len(datasets)} datasets\n")

    # Mettack + smoothing on subset (expensive)
    mettack_datasets = {"Cora", "Citeseer", "WikiCS"}

    all_results = {name: [] for name, _ in datasets}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for name, data in datasets:
            do_met = name in mettack_datasets
            do_smooth = name in mettack_datasets
            r = run_single_seed(name, data, seed, device,
                                run_mettack=do_met, run_smoothing=do_smooth)
            all_results[name].append(r)
            print(f"  {name}: acc={r['test_acc']:.3f} tight={r['constr_tight']:.3f} "
                  f"rho={r['rho']:.3f}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # ================================================================
    # TABLE 1: Cross-domain adversarial analysis (mean ± std)
    # ================================================================
    print("=" * 90)
    print("TABLE 1: Cross-Domain Adversarial Analysis (10 seeds)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'Acc':>12} {'rho':>12} {'Tight':>12} {'AtkAdv':>12} {'eps_crit':>12} {'Cert%':>12}")
    print("-" * 90)
    for name, _ in datasets:
        rs = all_results[name]
        print(f"{name:<12} {agg([r['test_acc'] for r in rs]):>12} "
              f"{agg([r['rho'] for r in rs]):>12} "
              f"{agg([r['constr_tight'] for r in rs]):>12} "
              f"{agg([r['atk_adv'] for r in rs]):>12} "
              f"{agg([r['eps_crit'] for r in rs]):>12} "
              f"{agg_pct([r['det_coverage'] for r in rs]):>12}")
    print()

    # ================================================================
    # TABLE 2: Mettack comparison (mean ± std)
    # ================================================================
    print("=" * 90)
    print("TABLE 2: IFT vs Mettack Attack (10 seeds)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'IFT tau':>12} {'Met tau':>12} {'IFT dmg':>12} {'Met dmg':>12} {'IFT wins':>10}")
    print("-" * 90)
    for name in mettack_datasets:
        rs = all_results[name]
        ift_taus = [r.get("ift_tau") for r in rs]
        met_taus = [r.get("met_tau") for r in rs]
        ift_dmgs = [r.get("ift_dmg_k1") for r in rs]
        met_dmgs = [r.get("met_dmg_k1") for r in rs]
        wins = sum(1 for i, m in zip(ift_dmgs, met_dmgs) if i is not None and m is not None and i > m)
        total = sum(1 for i, m in zip(ift_dmgs, met_dmgs) if i is not None and m is not None)
        print(f"{name:<12} {agg(ift_taus):>12} {agg(met_taus):>12} "
              f"{agg(ift_dmgs):>12} {agg(met_dmgs):>12} {wins}/{total}:>10")
    print()

    # ================================================================
    # TABLE 3: Certificate comparison (mean ± std)
    # ================================================================
    print("=" * 90)
    print("TABLE 3: Deterministic vs Smoothing Certificates (10 seeds)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'Det cov':>12} {'Smooth cov':>12} {'Det med_r':>12} {'Smooth med_r':>14} {'Radius ratio':>13}")
    print("-" * 90)
    for name in mettack_datasets:
        rs = all_results[name]
        det_covs = [r.get("det_coverage") for r in rs]
        smooth_covs = [r.get("smooth_coverage") for r in rs]
        det_rs = [r.get("det_med_r") for r in rs]
        smooth_rs = [r.get("smooth_med_r") for r in rs]
        ratios = [d / s if s and s > 1e-6 else None for d, s in zip(det_rs, smooth_rs)]
        print(f"{name:<12} {agg_pct(det_covs):>12} {agg_pct(smooth_covs):>12} "
              f"{agg(det_rs):>12} {agg(smooth_rs):>14} {agg(ratios):>13}x")
    print()

    # Save raw results for the paper
    results_path = Path("docs/multi_seed_results.md")
    with open(results_path, "w") as f:
        f.write("# Multi-Seed Experiment Results (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")

        f.write("## Table 1: Cross-Domain Adversarial Analysis\n\n")
        f.write(f"| Dataset | Acc | ρ | Tightness | AtkAdv | ε_crit | Cert% |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for name, _ in datasets:
            rs = all_results[name]
            f.write(f"| {name} | {agg([r['test_acc'] for r in rs])} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['atk_adv'] for r in rs])} "
                    f"| {agg([r['eps_crit'] for r in rs])} "
                    f"| {agg_pct([r['det_coverage'] for r in rs])} |\n")

        f.write("\n## Table 2: IFT vs Mettack\n\n")
        f.write(f"| Dataset | IFT τ | Mettack τ | IFT dmg (k=1) | Met dmg (k=1) |\n")
        f.write(f"|---|---|---|---|---|\n")
        for name in mettack_datasets:
            rs = all_results[name]
            f.write(f"| {name} "
                    f"| {agg([r.get('ift_tau') for r in rs])} "
                    f"| {agg([r.get('met_tau') for r in rs])} "
                    f"| {agg([r.get('ift_dmg_k1') for r in rs])} "
                    f"| {agg([r.get('met_dmg_k1') for r in rs])} |\n")

        f.write("\n## Table 3: Certificates\n\n")
        f.write(f"| Dataset | Det coverage | Smooth coverage | Det median r | Smooth median r |\n")
        f.write(f"|---|---|---|---|---|\n")
        for name in mettack_datasets:
            rs = all_results[name]
            f.write(f"| {name} "
                    f"| {agg_pct([r.get('det_coverage') for r in rs])} "
                    f"| {agg_pct([r.get('smooth_coverage') for r in rs])} "
                    f"| {agg([r.get('det_med_r') for r in rs])} "
                    f"| {agg([r.get('smooth_med_r') for r in rs])} |\n")

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
