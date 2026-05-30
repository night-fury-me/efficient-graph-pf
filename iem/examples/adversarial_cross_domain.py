"""Adversarial Equilibrium Theory — cross-domain validation.

Runs the full adversarial analysis (Theorems 1-3 + Proposition 1) on
all available domains: Cora, Citeseer, Pubmed, Amazon Photo, WikiCS,
and ContractiveGCN-PF (power flow).

Produces a unified summary table for the paper.

Usage:
    .venv/bin/python -m iem.examples.adversarial_cross_domain
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    certified_shift_bound,
    critical_perturbation_budget,
    extract_ego_subgraph,
    extract_W_spectral_norm,
    nonnormality_index,
    optimal_structural_attack,
    per_node_robust_radius,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


# ---------------------------------------------------------------------------
# Data loaders (reuse from existing examples)
# ---------------------------------------------------------------------------

def load_cora(device):
    _download_cora(Path("datasets/cora"))
    return _load_cora(Path("datasets/cora"))


def load_citeseer(device):
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    return _load_planetoid("citeseer", Path("datasets/citeseer"))


def load_pubmed(device):
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    return _load_planetoid("pubmed", Path("datasets/pubmed"))


def load_amazon(device):
    from iem.examples.ignn_amazon import _load_amazon
    return _load_amazon(Path("datasets/amazon_photo"))


def load_wikics(device):
    from iem.examples.ignn_wikics import _load_wikics
    return _load_wikics(Path("datasets/wikics"))


# ---------------------------------------------------------------------------
# Run adversarial analysis on one dataset
# ---------------------------------------------------------------------------

def run_adversarial_on_dataset(name: str, data: dict, device, n_epochs: int = 100):
    """Train IGNN + run full adversarial analysis. Returns summary dict."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}: N={data['N']}, feat={data['n_features']}, classes={data['n_classes']}", flush=True)
    print(f"{'='*70}", flush=True)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for ep in range(1, n_epochs + 1):
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
    print(f"  test_acc={test_acc:.3f}, params={n_params:,}", flush=True)

    # --- 50-node subgraph via BFS ---
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    S_size = len(idx)

    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    Z_sub = Z_star[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    # Reconverge to true subgraph fixed point
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new
    logits_sub = model.head(Z_sub)
    labels_sub = y[idx]

    n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
    print(f"  Subgraph: {S_size} nodes, {n_edges} edges", flush=True)

    # --- Spectral radius ---
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)
    print(f"  rho={rho:.4f}", flush=True)

    # --- Theorem 1: Certified bound ---
    t0 = time.time()
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    bound = certified_shift_bound(S, rho, epsilon=0.01)
    t_s = time.time() - t0
    print(f"  sigma_1(S)={bound['sigma_1']:.2f} (computed in {t_s:.1f}s)", flush=True)

    # --- Bound tightness ---
    tightness = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.001, 0.01],
    )
    tight_ratio = tightness[1]["constr_tightness"] if len(tightness) > 1 else tightness[0]["constr_tightness"]
    atk_adv = tightness[1]["attack_advantage"] if len(tightness) > 1 else tightness[0]["attack_advantage"]
    print(f"  tightness={tight_ratio:.3f}, attack_advantage={atk_adv:.1f}x", flush=True)

    # --- Theorem 2: Vulnerability spectrum ---
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    eff_dim = attack["effective_adversarial_dim"]

    # --- Theorem 3: Critical budget ---
    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = 1.0
    budget = critical_perturbation_budget(rho, W_norm)
    eps_crit = budget["epsilon_crit"]
    print(f"  eps_crit={eps_crit:.4f}", flush=True)

    # --- Proposition 1: Per-node certificates ---
    node_certs = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    frac_cert = node_certs["frac_correct_and_certified"]
    med_radius = node_certs["median_radius"]
    print(f"  certified={frac_cert:.0%}, median_r={med_radius:.4f}", flush=True)

    # --- Non-normality ---
    nn = nonnormality_index(J_z, rho)
    eta = nn["nonnormality_index"]

    return {
        "name": name,
        "N": data["N"],
        "n_edges_full": int((A_hat.abs() > 1e-10).sum() - data["N"]) // 2,
        "test_acc": test_acc,
        "sub_nodes": S_size,
        "sub_edges": n_edges,
        "rho": rho,
        "sigma_1": bound["sigma_1"],
        "tight_ratio": tight_ratio,
        "atk_adv": atk_adv,
        "eff_dim": eff_dim,
        "eps_crit": eps_crit,
        "frac_cert": frac_cert,
        "med_radius": med_radius,
        "eta": eta,
    }


# ---------------------------------------------------------------------------
# Power flow domain (ContractiveGCN-PF)
# ---------------------------------------------------------------------------

def run_adversarial_pf(device):
    """Run adversarial analysis on ContractiveGCN-PF + HVN."""
    print(f"\n{'='*70}", flush=True)
    print(f"  ContractiveGCN-PF (Power Flow / HVN)", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        from iem.examples.contractive_pf import ContractiveGCN_PF
    except ImportError:
        print("  SKIP: contractive_pf not available", flush=True)
        return None

    from data_loading.collate import collate_blockdiag
    from data_loading.dataset import ChanghunDataset
    from torch.utils.data import DataLoader
    import models  # noqa: register

    ds_path = Path("./datasets/HVN_stratified_1500.parquet")
    if not ds_path.exists():
        print("  SKIP: HVN dataset not found", flush=True)
        return None

    ds = ChanghunDataset([str(ds_path)], per_unit=True, device=device)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(50):
        model.train()
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()

    # Use first small grid
    batch = next(iter(loader))
    with torch.no_grad():
        V_pred, ctx_pf = model(
            batch["bus_type"].to(device), batch["Lines_connected"].to(device),
            None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device), batch["V_start"].to(device),
            batch["sizes"].to(device),
        )
    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    X_proj = ctx_pf["X_proj"]
    N = int(batch["sizes"][0].item())
    ctx_sub = {"A_hat": A_hat[:N, :N], "X_proj": X_proj[:N]}
    Z_sub = Z_star[:N]

    # Reconverge
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)
    n_edges = int((ctx_sub["A_hat"].abs() > 1e-10).sum() - N) // 2
    print(f"  N={N}, edges={n_edges}, rho={rho:.4f}", flush=True)

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    bound = certified_shift_bound(S, rho, epsilon=0.01)
    print(f"  sigma_1(S)={bound['sigma_1']:.2f}", flush=True)

    tightness = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.001, 0.01],
    )
    tight_ratio = tightness[-1]["constr_tightness"]
    atk_adv = tightness[-1]["attack_advantage"]

    attack = optimal_structural_attack(S, ctx_sub["A_hat"], epsilon=0.01)

    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = 1.0
    budget = critical_perturbation_budget(rho, W_norm)

    nn = nonnormality_index(J_z, rho)

    print(f"  tight={tight_ratio:.3f}, atk_adv={atk_adv:.1f}x, eps_crit={budget['epsilon_crit']:.4f}", flush=True)

    return {
        "name": "PF (HVN)",
        "N": N,
        "n_edges_full": n_edges,
        "test_acc": float("nan"),
        "sub_nodes": N,
        "sub_edges": n_edges,
        "rho": rho,
        "sigma_1": bound["sigma_1"],
        "tight_ratio": tight_ratio,
        "atk_adv": atk_adv,
        "eff_dim": attack["effective_adversarial_dim"],
        "eps_crit": budget["epsilon_crit"],
        "frac_cert": float("nan"),
        "med_radius": float("nan"),
        "eta": nn["nonnormality_index"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = [
        ("Cora (Citations)", load_cora),
        ("Citeseer (CS)", load_citeseer),
        ("Pubmed (Biomed)", load_pubmed),
        ("Amazon Photo", load_amazon),
        ("WikiCS (Encylopedia)", load_wikics),
    ]

    results = []

    for name, loader_fn in loaders:
        try:
            data = loader_fn(device)
            r = run_adversarial_on_dataset(name, data, device)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}", flush=True)

    # Power flow
    try:
        r_pf = run_adversarial_pf(device)
        if r_pf:
            results.append(r_pf)
    except Exception as e:
        print(f"\n  ERROR on PF: {e}", flush=True)

    # --- Summary table ---
    print(f"\n\n{'='*100}", flush=True)
    print("ADVERSARIAL EQUILIBRIUM THEORY — CROSS-DOMAIN SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Dataset':<22} {'N':>6} {'rho':>6} {'sigma1':>8} {'tight':>6} {'atk_adv':>8} "
          f"{'eps_crit':>8} {'cert%':>6} {'med_r':>8} {'eta':>5}", flush=True)
    print("-" * 100, flush=True)
    for r in results:
        cert_s = f"{r['frac_cert']:.0%}" if not (r['frac_cert'] != r['frac_cert']) else "N/A"
        med_s = f"{r['med_radius']:.4f}" if not (r['med_radius'] != r['med_radius']) else "N/A"
        print(f"{r['name']:<22} {r['N']:>6} {r['rho']:>6.3f} {r['sigma_1']:>8.2f} "
              f"{r['tight_ratio']:>6.3f} {r['atk_adv']:>7.1f}x {r['eps_crit']:>8.4f} "
              f"{cert_s:>6} {med_s:>8} {r['eta']:>5.2f}", flush=True)
    print(f"{'='*100}\n", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
