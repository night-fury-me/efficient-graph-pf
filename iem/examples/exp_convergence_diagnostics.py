"""P3 Experiment: DEQ convergence diagnostics across all datasets.

Reports convergence statistics for the IGNN fixed-point iteration:
iteration count, final residual, convergence rate. Validates that
Anderson acceleration converges reliably across all 9 datasets.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_convergence_diagnostics
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

from iem.adversarial import extract_ego_subgraph, nonnormality_index
from iem.certify import spectral_radius
from iem.ift import compute_jacobian
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_convergence(model, X, A_hat, ctx, max_iter=200, tol=1e-6):
    """Measure fixed-point iteration convergence statistics."""
    N = X.shape[0]
    Z = torch.zeros(N, model.hidden, device=X.device)
    residuals = []

    with torch.no_grad():
        for k in range(max_iter):
            Z_new = model.operator(Z, ctx)
            residual = float((Z_new - Z).norm())
            residuals.append(residual)
            if residual < tol:
                break
            Z = Z_new

    converged = residuals[-1] < tol
    n_iter = len(residuals)

    # Convergence rate: geometric decay factor
    if n_iter >= 3:
        rates = [residuals[i+1] / max(residuals[i], 1e-15) for i in range(min(n_iter-1, 10))]
        avg_rate = np.mean([r for r in rates if r < 10])
    else:
        avg_rate = 0.0

    # Verify true fixed point
    fp_residual = float((model.operator(Z_new, ctx) - Z_new).norm())

    return {
        "converged": converged,
        "n_iter": n_iter,
        "final_residual": residuals[-1],
        "fp_residual": fp_residual,
        "avg_conv_rate": avg_rate,
    }


def run_single_graph(name, data, seed, device):
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

    # Full-graph convergence
    conv = measure_convergence(model, X, A_hat, ctx)

    # Subgraph convergence + non-normality
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    # Non-normality index
    J_z = compute_jacobian(F_z, Z_sub)
    nn = nonnormality_index(J_z, rho)

    return {
        "converged": conv["converged"],
        "n_iter": conv["n_iter"],
        "final_residual": conv["final_residual"],
        "fp_residual": conv["fp_residual"],
        "avg_conv_rate": conv["avg_conv_rate"],
        "rho": rho,
        "eta": nn["nonnormality_index"],
        "resolvent_norm": nn["resolvent_norm"],
    }


def run_single_ieee(case_name, ds_path, seed, device):
    set_seed(seed)

    if not Path(ds_path).exists():
        return None

    import models  # noqa: register
    from iem.examples.contractive_pf import ContractiveGCN_PF
    from data_loading.collate import collate_blockdiag
    from data_loading.dataset import ChanghunDataset
    from torch.utils.data import DataLoader, Subset

    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(30):
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

    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)))
    with torch.no_grad():
        V_pred, ctx_pf = model(
            batch["bus_type"].to(device), batch["Lines_connected"].to(device),
            None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device), batch["V_start"].to(device),
            batch["sizes"].to(device),
        )

    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    N = int(batch["sizes"][0].item())
    A_sub = A_hat[:N, :N]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx_pf["X_proj"][:N]}
    Z_sub = Z_star[:N]

    # Measure convergence from scratch
    Z = torch.zeros_like(Z_sub)
    residuals = []
    with torch.no_grad():
        for k in range(200):
            Z_new = model.operator(Z, ctx_sub)
            residual = float((Z_new - Z).norm())
            residuals.append(residual)
            if residual < 1e-6:
                break
            Z = Z_new

    converged = residuals[-1] < 1e-6
    n_iter = len(residuals)

    if n_iter >= 3:
        rates = [residuals[i+1] / max(residuals[i], 1e-15) for i in range(min(n_iter-1, 10))]
        avg_rate = np.mean([r for r in rates if r < 10])
    else:
        avg_rate = 0.0

    fp_residual = float((model.operator(Z_new, ctx_sub) - Z_new).norm())

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)

    # Reconverge for rho
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new2 = model.operator(Z, ctx_sub)
            if (Z_new2 - Z).norm() < 1e-7:
                break
            Z = Z_new2

    rho = spectral_radius(
        lambda z, _c=ctx_sub: model.operator(z.reshape(Z_new2.shape), _c).reshape(-1),
        Z_new2,
    )

    J_z = compute_jacobian(
        lambda z, _c=ctx_sub: model.operator(z.reshape(Z_new2.shape), _c).reshape(-1),
        Z_new2,
    )
    nn = nonnormality_index(J_z, rho)

    return {
        "converged": converged,
        "n_iter": n_iter,
        "final_residual": residuals[-1],
        "fp_residual": fp_residual,
        "avg_conv_rate": avg_rate,
        "rho": rho,
        "eta": nn["nonnormality_index"],
        "resolvent_norm": nn["resolvent_norm"],
    }


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def agg_int(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.0f}±{s:.0f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_amazon import _load_amazon
    from iem.examples.ignn_wikics import _load_wikics

    graph_datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("Pubmed", _load_planetoid("pubmed", Path("datasets/pubmed"))),
        ("Amazon", _load_amazon(Path("datasets/amazon_photo"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    ieee_cases = [
        ("case14", "datasets/IEEE_case14_2000.parquet"),
        ("case30", "datasets/IEEE_case30_2000.parquet"),
        ("case57", "datasets/IEEE_case57_2000.parquet"),
        ("case118", "datasets/IEEE_case118_2000.parquet"),
    ]

    all_results = {}

    # Graph benchmarks
    for name, data in graph_datasets:
        all_results[name] = []
        for seed_idx, seed in enumerate(SEEDS):
            r = run_single_graph(name, data, seed, device)
            all_results[name].append(r)
            if seed_idx == 0:
                print(f"  {name}: iter={r['n_iter']} rho={r['rho']:.3f} eta={r['eta']:.2f} "
                      f"conv={r['converged']}", flush=True)
        print(f"  {name}: {len(SEEDS)} seeds done", flush=True)

    # IEEE cases
    for case_name, ds_path in ieee_cases:
        all_results[case_name] = []
        for seed_idx, seed in enumerate(SEEDS):
            r = run_single_ieee(case_name, ds_path, seed, device)
            if r:
                all_results[case_name].append(r)
        if all_results[case_name]:
            print(f"  {case_name}: {len(all_results[case_name])} seeds done", flush=True)
        else:
            print(f"  {case_name}: SKIP", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Table
    print("=" * 110)
    print("DEQ CONVERGENCE DIAGNOSTICS (10 seeds)")
    print("=" * 110)
    print(f"{'Dataset':<12} {'Converged':>10} {'Iterations':>12} {'FP residual':>14} "
          f"{'Conv rate':>12} {'rho':>12} {'eta':>10}")
    print("-" * 110)

    all_names = [n for n, _ in graph_datasets] + [n for n, _ in ieee_cases]
    for name in all_names:
        rs = all_results.get(name, [])
        if not rs:
            continue
        conv_pct = f"{np.mean([r['converged'] for r in rs])*100:.0f}%"
        print(f"{name:<12} {conv_pct:>10} {agg_int([r['n_iter'] for r in rs]):>12} "
              f"{agg([r['fp_residual'] for r in rs], '.1e'):>14} "
              f"{agg([r['avg_conv_rate'] for r in rs]):>12} "
              f"{agg([r['rho'] for r in rs]):>12} "
              f"{agg([r['eta'] for r in rs], '.2f'):>10}")

    # Save
    results_path = Path("docs/exp_convergence_diagnostics_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# DEQ Convergence Diagnostics (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Dataset | Converged | Iterations | FP residual | Conv rate | ρ | η |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name in all_names:
            rs = all_results.get(name, [])
            if not rs:
                continue
            conv_pct = f"{np.mean([r['converged'] for r in rs])*100:.0f}%"
            f.write(f"| {name} | {conv_pct} "
                    f"| {agg_int([r['n_iter'] for r in rs])} "
                    f"| {agg([r['fp_residual'] for r in rs], '.1e')} "
                    f"| {agg([r['avg_conv_rate'] for r in rs])} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {agg([r['eta'] for r in rs], '.2f')} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
