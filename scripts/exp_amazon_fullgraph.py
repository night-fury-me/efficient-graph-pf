"""C2 follow-up: Full-graph matrix-free analysis on Amazon Photo.

Tests whether the negative τ is a subgraph artifact:
  - Subgraph (50 nodes): ||Â_sub||₂ = 0.20, κ_sub = 0.14 → τ = -0.15
  - Full graph (7650 nodes): ||Â||₂ = 1.00, κ_full = ? → τ = ?

Full-graph edge_vulnerability via matrix-free pipeline (Neumann + JVP).
Discrete ground truth sampled on 200 random edges (full brute-force is
infeasible at 119K edges).

Usage:
    .venv/bin/python scripts/exp_amazon_fullgraph.py
"""

from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.examples.ignn_cora import IGNN
from iem.examples.ignn_amazon import _load_amazon
from iem.adversarial import extract_ego_subgraph
from iem.scalable import ScalableSensitivity

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]  # 10 seeds
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
N_SAMPLE_EDGES = 200
N_TOP_STRATA = 100   # stratified top-K AEGIS edges
N_RANDOM_STRATA = 100  # plus random fill


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def reconverge(model, Z, ctx, max_iter=200):
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7: break
            Z = Z_new
    return Z_new


def train_ignn(data, device, seed):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        lo, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(lo[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad(); loss.backward(); optim.step()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = model(X, A_hat)
                va = float((lv.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model


def run_single(data, seed, device):
    set_seed(seed)
    model = train_ignn(data, device, seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    N = A_hat.shape[0]
    A_hat_sn = float(torch.linalg.svdvals(A_hat)[0])
    W_sn = float(torch.linalg.svdvals(model.W.weight.detach())[0])
    kappa_full = A_hat_sn * W_sn

    # ---- Full-graph matrix-free analysis ----
    print("  full-graph matrix-free...", end="", flush=True)
    t0 = time.time()
    op = ScalableSensitivity(
        lambda z, c: model.operator(z, c),
        Z_star, ctx,
        neumann_tol=1e-5,
    )
    rho_full = op.rho

    # Get vulnerability for ALL edges (expensive but feasible)
    vulns_full = op.edge_vulnerability()
    t_full = time.time() - t0
    print(f" {t_full:.0f}s, {len(vulns_full)} edges, rho={rho_full:.3f}", flush=True)

    # Build vuln dict for lookup
    vuln_dict = {(i, j): v for i, j, v in vulns_full}

    # STRATIFIED SAMPLING: top-K by AEGIS v_ij + K random from the rest
    rng = random.Random(seed)
    vulns_sorted = sorted(vulns_full, key=lambda r: -r[2])  # descending
    top_strata = vulns_sorted[:N_TOP_STRATA]
    rest_pool = vulns_sorted[N_TOP_STRATA:]
    random_strata = rng.sample(rest_pool, min(N_RANDOM_STRATA, len(rest_pool)))
    sample_edges = top_strata + random_strata

    # Brute-force discrete damage on stratified sample
    print(f"  brute-force on {len(sample_edges)} (top-{N_TOP_STRATA} + random-{N_RANDOM_STRATA}) edges...",
          end="", flush=True)
    t0 = time.time()
    aegis_scores = []
    aegis_weighted_scores = []  # A_hat[i,j] * v_ij (edge-weighted, first-order proxy)
    discrete_scores = []
    with torch.no_grad():
        for i, j, v in sample_edges:
            aegis_scores.append(v)
            a_ij = float(A_hat[i, j].item())
            aegis_weighted_scores.append(a_ij * v)
            A_pert = A_hat.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx, "A_hat": A_pert}
            Z_pert = reconverge(model, Z_star, ctx_pert)
            discrete_scores.append(float((Z_pert - Z_star).norm()))
    t_bf = time.time() - t0
    print(f" {t_bf:.0f}s", flush=True)

    # Three τ variants:
    #   (1) raw v_ij on stratified sample
    #   (2) A_hat[i,j] * v_ij on stratified sample (first-order edge-weighted)
    #   (3) raw v_ij on top-strata only (head of distribution)
    tau_full, _ = kendalltau(aegis_scores, discrete_scores)
    tau_full_weighted, _ = kendalltau(aegis_weighted_scores, discrete_scores)
    tau_top, _ = kendalltau(aegis_scores[:N_TOP_STRATA],
                             discrete_scores[:N_TOP_STRATA])
    print(f"  tau_strat_raw={tau_full:+.3f} tau_strat_wgt={tau_full_weighted:+.3f} "
          f"tau_top100={tau_top:+.3f}", flush=True)

    # P@10 on sampled edges
    k10 = min(10, len(sample_edges))
    gt_top = set(np.argsort(discrete_scores)[-k10:])
    ae_top = set(np.argsort(aegis_scores)[-k10:])
    p10_full = len(gt_top & ae_top) / k10

    # ---- Subgraph analysis (for comparison) ----
    print("  subgraph (50-node)...", end="", flush=True)
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)

    A_sub_sn = float(torch.linalg.svdvals(A_sub)[0])
    kappa_sub = A_sub_sn * W_sn

    from iem.adversarial import (
        _compute_structural_jacobian, structural_sensitivity_matrix,
        constrained_sensitivity_matrix, greedy_structural_attack,
    )
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

    sc_scores = [float(S_c[:, k].norm()) for k in range(len(edge_list))]
    bf_sub = greedy_structural_attack(model, Z_sub, ctx_sub)
    bf_dict = {(min(i,j), max(i,j)): s for i,j,s in bf_sub}
    bf_matched = [bf_dict.get((min(i,j), max(i,j)), 0.0) for i,j in edge_list]
    tau_sub, _ = kendalltau(sc_scores, bf_matched)
    print(f" tau_sub={tau_sub:+.3f}", flush=True)

    del model, S, S_c, J_z, J_A, op
    gc.collect(); torch.cuda.empty_cache()

    return {
        "seed": seed,
        "kappa_full": kappa_full, "kappa_sub": kappa_sub,
        "rho_full": rho_full,
        "A_hat_sn": A_hat_sn, "A_sub_sn": A_sub_sn, "W_sn": W_sn,
        "tau_full": tau_full,
        "tau_full_weighted": tau_full_weighted,
        "tau_top100": tau_top,
        "tau_sub": tau_sub,
        "p10_full": p10_full,
        "n_edges_full": len(vulns_full),
        "n_edges_sampled": len(sample_edges),
        "t_fullgraph": t_full, "t_bf": t_bf,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data = _load_amazon(Path("datasets/amazon_photo"))
    print(f"Amazon Photo: N={data['N']}\n")

    rows = []
    for si, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({si+1}/{len(SEEDS)}) ===")
        r = run_single(data, seed, device)
        rows.append(r)
        print(f"  RESULT: kappa_full={r['kappa_full']:.3f} vs kappa_sub={r['kappa_sub']:.3f}")
        print(f"          tau_full={r['tau_full']:+.3f} vs tau_sub={r['tau_sub']:+.3f}")
        print(f"          rho_full={r['rho_full']:.3f}")
        print()

    print("=" * 80)
    print("AMAZON PHOTO: FULL-GRAPH vs SUBGRAPH COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Full graph':>15} {'50-node sub':>15}")
    print("-" * 50)
    print(f"{'||Â||₂':<20} {np.mean([r['A_hat_sn'] for r in rows]):>15.3f} {np.mean([r['A_sub_sn'] for r in rows]):>15.3f}")
    print(f"{'κ':<20} {np.mean([r['kappa_full'] for r in rows]):>15.3f} {np.mean([r['kappa_sub'] for r in rows]):>15.3f}")
    print(f"{'ρ (spectral radius)':<20} {np.mean([r['rho_full'] for r in rows]):>15.3f} {'---':>15}")
    print(f"{'τ vs discrete':<20} {np.mean([r['tau_full'] for r in rows]):>+14.3f} {np.mean([r['tau_sub'] for r in rows]):>+14.3f}")
    print(f"{'P@10':<20} {np.mean([r['p10_full'] for r in rows]):>15.3f} {'---':>15}")
    print(f"{'|E|':<20} {np.mean([r['n_edges_full'] for r in rows]):>15.0f} {'~50':>15}")
    print(f"{'Time (s)':<20} {np.mean([r['t_fullgraph'] for r in rows]):>15.0f} {'<1':>15}")

    # Write the stratified-sample tau values to a dedicated CSV
    import csv as _csv
    out = Path("results/revision_R2/amazon_fullgraph_stratified.csv")
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure output dir exists (missing in isolated job dirs)
    new_rows = []
    for r in rows:
        new_rows.append({
            "dataset": "AmazonPhoto",
            "seed": r["seed"],
            "n_edges_full": r["n_edges_full"],
            "n_top_strata": N_TOP_STRATA,
            "n_random_strata": N_RANDOM_STRATA,
            "tau_strat_raw":       r["tau_full"],
            "tau_strat_weighted":  r["tau_full_weighted"],
            "tau_top100_only":     r["tau_top100"],
            "tau_sub":             r["tau_sub"],
            "p10_full":            r["p10_full"],
            "kappa_full":          r["kappa_full"],
            "kappa_sub":           r["kappa_sub"],
            "rho_full":            r["rho_full"],
            "t_fullgraph":         r["t_fullgraph"],
            "t_bf":                r["t_bf"],
        })
    write_header = not out.exists()
    fields = ["dataset", "seed", "n_edges_full", "n_top_strata", "n_random_strata",
              "tau_strat_raw", "tau_strat_weighted", "tau_top100_only", "tau_sub",
              "p10_full", "kappa_full", "kappa_sub", "rho_full",
              "t_fullgraph", "t_bf"]
    with out.open("a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        for nr in new_rows:
            w.writerow(nr)
    print(f"\nAppended {len(new_rows)} Amazon Photo rows to {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
