"""P0 Experiment: Smoothing certificate comparison at standard sigma values.

Compares AEGIS deterministic certificates against randomized smoothing at
sigma={0.10, 0.25, 0.50} with 1000 Monte Carlo samples each (standard
practice per Bojchevski et al., 2020). Reports coverage, median radius,
and base accuracy under noise.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_smoothing_sweep
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    extract_ego_subgraph,
    per_node_robust_radius,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
SIGMAS = [0.10, 0.25, 0.50]
N_SAMPLES = 1000


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoothing_certificate_with_accuracy(
    model,
    z_clean: torch.Tensor,
    ctx: dict,
    labels: torch.Tensor,
    sigma: float,
    n_samples: int,
    alpha: float = 0.001,
    A_key: str = "A_hat",
) -> dict:
    """Randomized smoothing with accuracy tracking under noise."""
    from scipy.stats import norm as scipy_norm
    from scipy.stats import beta as beta_dist

    A = ctx[A_key]
    N = labels.shape[0]
    correct_counts = torch.zeros(N)
    clean_preds = model.head(z_clean).argmax(dim=1).cpu()

    with torch.no_grad():
        for _ in range(n_samples):
            dA = torch.randn_like(A) * sigma
            dA = (dA + dA.T) / 2
            dA.fill_diagonal_(0)
            ctx_pert = {**ctx, A_key: A + dA}
            Z = z_clean.clone()
            for _ in range(100):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
            logits_pert = model.head(Z_new)
            pred = logits_pert.argmax(dim=1).cpu()
            correct_counts += (pred == labels.cpu()).float()

    p_A = correct_counts / n_samples

    # Smoothed accuracy: fraction that maintain correct prediction >50% of time
    smoothed_acc = float((p_A > 0.5).float().mean())

    radii = torch.zeros(N)
    for v in range(N):
        if p_A[v] > 0.5:
            k = int(p_A[v].item() * n_samples)
            if k >= n_samples:
                p_lower = alpha ** (1.0 / n_samples)
            else:
                p_lower = beta_dist.ppf(alpha / 2, k, n_samples - k + 1)
            p_lower = max(p_lower, 0.5 + 1e-6)
            p_lower = min(p_lower, 1.0 - 1e-10)
            radii[v] = sigma * scipy_norm.ppf(p_lower)

    frac_cert = float((radii > 1e-6).float().mean())
    nontrivial = radii[radii > 1e-6]

    return {
        "radii": radii,
        "frac_certified": frac_cert,
        "median_radius": float(nontrivial.median()) if len(nontrivial) > 0 else 0.0,
        "smoothed_accuracy": smoothed_acc,
        "sigma": sigma,
    }


def run_single(name, data, seed, device):
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float((logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
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

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    # Deterministic certificates
    det = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    det_nontrivial = det["radii"][det["radii"] > 1e-6]

    result = {
        "det_coverage": det["frac_correct_and_certified"],
        "det_median_r": float(det_nontrivial.median()) if len(det_nontrivial) > 0 else 0.0,
    }

    # Smoothing at each sigma
    for sigma in SIGMAS:
        smooth = smoothing_certificate_with_accuracy(
            model, Z_sub, ctx_sub, labels_sub,
            sigma=sigma, n_samples=N_SAMPLES,
        )
        result[f"smooth_{sigma}_coverage"] = smooth["frac_certified"]
        result[f"smooth_{sigma}_median_r"] = smooth["median_radius"]
        result[f"smooth_{sigma}_acc"] = smooth["smoothed_accuracy"]

    return result


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def agg_pct(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m*100:.0f}±{s*100:.0f}%"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    all_results = {name: [] for name, _ in datasets}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for name, data in datasets:
            r = run_single(name, data, seed, device)
            all_results[name].append(r)
            print(f"  {name}: det_r={r['det_median_r']:.4f} "
                  f"sm025_r={r['smooth_0.25_median_r']:.4f} "
                  f"sm050_r={r['smooth_0.5_median_r']:.4f}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Table
    print("=" * 120)
    print("SMOOTHING CERTIFICATE SWEEP (10 seeds, 1000 MC samples)")
    print("=" * 120)
    print(f"{'Dataset':<12} {'Det rad':>10} ", end="")
    for sigma in SIGMAS:
        print(f"{'σ='+str(sigma)+' rad':>12} {'σ='+str(sigma)+' acc':>12} ", end="")
    print()
    print("-" * 120)
    for name, _ in datasets:
        rs = all_results[name]
        print(f"{name:<12} {agg([r['det_median_r'] for r in rs]):>10} ", end="")
        for sigma in SIGMAS:
            print(f"{agg([r[f'smooth_{sigma}_median_r'] for r in rs]):>12} "
                  f"{agg_pct([r[f'smooth_{sigma}_acc'] for r in rs]):>12} ", end="")
        print()

    # Save
    results_path = Path("docs/exp_smoothing_sweep_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Smoothing Certificate Sweep (10 seeds, 1000 MC samples)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Dataset | Det rad | ")
        for sigma in SIGMAS:
            f.write(f"σ={sigma} rad | σ={sigma} acc | ")
        f.write("\n|---|---|")
        for _ in SIGMAS:
            f.write("---|---|")
        f.write("\n")
        for name, _ in datasets:
            rs = all_results[name]
            f.write(f"| {name} | {agg([r['det_median_r'] for r in rs])} | ")
            for sigma in SIGMAS:
                f.write(f"{agg([r[f'smooth_{sigma}_median_r'] for r in rs])} | "
                        f"{agg_pct([r[f'smooth_{sigma}_acc'] for r in rs])} | ")
            f.write("\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
