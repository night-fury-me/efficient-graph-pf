"""Baseline comparisons for Adversarial Equilibrium Theory.

Three comparisons addressing reviewer concern of missing baselines:

1. IFT-SVD attack vs greedy brute-force (strongest single-edge baseline)
   → measures: Kendall tau between IFT vulnerability ranking and brute-force ranking

2. Deterministic IFT certificates vs randomized smoothing certificates
   → measures: coverage, median radius, deterministic vs probabilistic

3. IFT N-1 ranking vs DC-PF PTDF/LODF on power flow
   → measures: Kendall tau against brute-force ground truth

Usage:
    .venv/bin/python -m iem.examples.adversarial_baselines
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    greedy_structural_attack,
    optimal_structural_attack,
    per_node_robust_radius,
    randomized_smoothing_certificate,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


def run_attack_comparison(name, model, Z_sub, ctx_sub, S, A_sub, idx, y, device):
    """Compare IFT-SVD attack ranking vs greedy brute-force ranking."""
    print(f"\n--- Baseline 1: Attack ranking ({name}) ---", flush=True)

    # IFT vulnerability spectrum
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    ift_edges = {(i, j): v for i, j, v in attack["all_edge_vulnerabilities"]}

    # Greedy brute-force (ground truth)
    t0 = time.time()
    bf = greedy_structural_attack(model, Z_sub, ctx_sub)
    bf_time = time.time() - t0

    # IFT time (already computed via S)
    ift_time = 0.0  # S computation time is shared

    # Match edges and compute correlation
    common_edges = []
    for i, j, bf_shift in bf:
        if (i, j) in ift_edges:
            common_edges.append((bf_shift, ift_edges[(i, j)]))
        elif (j, i) in ift_edges:
            common_edges.append((bf_shift, ift_edges[(j, i)]))

    if len(common_edges) >= 3:
        bf_vals = [x[0] for x in common_edges]
        ift_vals = [x[1] for x in common_edges]
        tau, p = kendalltau(bf_vals, ift_vals)

        k = min(5, len(common_edges))
        bf_top = set(range(k))
        ift_ranked = sorted(range(len(common_edges)), key=lambda i: common_edges[i][1], reverse=True)
        ift_top = set(ift_ranked[:k])
        top_k_agree = len(bf_top & ift_top) / k

        print(f"  Edges compared: {len(common_edges)}")
        print(f"  Kendall tau (IFT vs brute-force): {tau:+.3f} (p={p:.2e})")
        print(f"  Top-{k} agreement: {top_k_agree:.0%}")
        print(f"  Brute-force time: {bf_time:.1f}s")
        return {"tau": tau, "top_k": top_k_agree, "n_edges": len(common_edges), "bf_time": bf_time}
    else:
        print(f"  Too few common edges ({len(common_edges)})")
        return {"tau": None, "n_edges": len(common_edges)}


def run_certificate_comparison(name, model, Z_sub, ctx_sub, S, labels_sub, rho, device):
    """Compare deterministic IFT certificates vs randomized smoothing."""
    print(f"\n--- Baseline 2: Certificates ({name}) ---", flush=True)

    logits_sub = model.head(Z_sub)

    # Our deterministic certificates
    t0 = time.time()
    det = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    det_time = time.time() - t0

    # Randomized smoothing baseline
    t0 = time.time()
    smooth = randomized_smoothing_certificate(
        model, Z_sub, ctx_sub, labels_sub, sigma=0.01, n_samples=200,
    )
    smooth_time = time.time() - t0

    N = labels_sub.shape[0]
    det_radii = det["radii"]
    smooth_radii = smooth["radii"]

    det_cert = float((det_radii > 1e-6).float().mean())
    smooth_cert = smooth["frac_certified"]

    det_nontrivial = det_radii[det_radii > 1e-6]
    smooth_nontrivial = smooth_radii[smooth_radii > 1e-6]

    print(f"  {'':30} {'Deterministic (ours)':>20} {'Smoothing':>20}")
    print(f"  {'Coverage':30} {det_cert:>19.0%} {smooth_cert:>19.0%}")
    if len(det_nontrivial) > 0:
        print(f"  {'Median radius':30} {float(det_nontrivial.median()):>20.4f} "
              f"{float(smooth_nontrivial.median()) if len(smooth_nontrivial) > 0 else 0:>20.4f}")
        print(f"  {'Mean radius':30} {float(det_nontrivial.mean()):>20.4f} "
              f"{float(smooth_nontrivial.mean()) if len(smooth_nontrivial) > 0 else 0:>20.4f}")
    print(f"  {'Guarantee type':30} {'deterministic':>20} {'prob (1-alpha)':>20}")
    print(f"  {'Compute time':30} {det_time:>19.1f}s {smooth_time:>19.1f}s")

    # Correlation between the two sets of radii
    both_nz = (det_radii > 1e-6) & (smooth_radii > 1e-6)
    if both_nz.sum() >= 3:
        tau, p = kendalltau(det_radii[both_nz].numpy(), smooth_radii[both_nz].numpy())
        print(f"  Kendall tau (det vs smooth): {tau:+.3f} (p={p:.2e})")
    else:
        tau = None

    return {
        "det_coverage": det_cert,
        "smooth_coverage": smooth_cert,
        "det_median": float(det_nontrivial.median()) if len(det_nontrivial) > 0 else 0,
        "smooth_median": float(smooth_nontrivial.median()) if len(smooth_nontrivial) > 0 else 0,
        "tau": tau,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    attack_results = []
    cert_results = []

    for name, data in datasets:
        print(f"\n{'='*70}", flush=True)
        print(f"  {name}: N={data['N']}", flush=True)
        print(f"{'='*70}", flush=True)

        X = data["X"].to(device)
        A_hat = data["A_hat"].to(device)
        y = data["y"].to(device)

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
        print(f"  test_acc={acc:.3f}", flush=True)

        # Subgraph
        deg = A_hat.sum(dim=1)
        center = int(deg.argmax().item())
        neighbors = (A_hat[center] > 0).nonzero(as_tuple=True)[0]
        idx = neighbors[:50]
        S_size = len(idx)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

        # Reconverge to subgraph fixed point
        Z_sub = Z_star[idx].clone()
        with torch.no_grad():
            for _ in range(200):
                Z_new = model.operator(Z_sub, ctx_sub)
                if (Z_new - Z_sub).norm() < 1e-7:
                    break
                Z_sub = Z_new
        Z_sub = Z_new
        labels_sub = y[idx]

        def F_z(z):
            return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
        rho = spectral_radius(F_z, Z_sub)

        n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
        print(f"  Subgraph: {S_size} nodes, {n_edges} edges, rho={rho:.4f}", flush=True)

        # Compute S
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
        )

        # Baseline 1: Attack ranking
        r1 = run_attack_comparison(name, model, Z_sub, ctx_sub, S, A_sub, idx, y, device)
        r1["name"] = name
        attack_results.append(r1)

        # Baseline 2: Certificates
        r2 = run_certificate_comparison(name, model, Z_sub, ctx_sub, S, labels_sub, rho, device)
        r2["name"] = name
        cert_results.append(r2)

    # --- Summary ---
    print(f"\n\n{'='*70}", flush=True)
    print("BASELINE COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Attack ranking: IFT-SVD vs Greedy Brute-Force", flush=True)
    print(f"  {'Dataset':<15} {'Edges':>6} {'Kendall tau':>12} {'Top-5':>8} {'BF time':>8}", flush=True)
    for r in attack_results:
        tau_s = f"{r['tau']:+.3f}" if r.get('tau') is not None else "N/A"
        top_s = f"{r.get('top_k', 0):.0%}" if r.get('top_k') is not None else "N/A"
        bf_s = f"{r.get('bf_time', 0):.1f}s" if r.get('bf_time') else "N/A"
        print(f"  {r['name']:<15} {r['n_edges']:>6} {tau_s:>12} {top_s:>8} {bf_s:>8}", flush=True)

    print(f"\nCertificate comparison: Deterministic (ours) vs Randomized Smoothing", flush=True)
    print(f"  {'Dataset':<15} {'Det cov':>8} {'Smooth cov':>10} {'Det med_r':>10} {'Smooth med_r':>12}", flush=True)
    for r in cert_results:
        print(f"  {r['name']:<15} {r['det_coverage']:>7.0%} {r['smooth_coverage']:>9.0%} "
              f"{r['det_median']:>10.4f} {r['smooth_median']:>12.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
