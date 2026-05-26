"""Compute tightness degradation for Pubmed and Amazon Photo (missing from Table II).

Measures tightness = actual_shift / predicted_shift at eps = 0.01, 0.05, 0.10, 0.20.
Also computes weight-corrected tau (Proposition 3 verification).

Usage:
    .venv/bin/python scripts/exp_tightness_expansion.py
"""
from __future__ import annotations

import csv
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    optimal_structural_attack,
    structural_sensitivity_matrix,
    extract_ego_subgraph,
    greedy_structural_attack,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
EPS_VALUES = [0.01, 0.05, 0.10, 0.20]


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_tightness(ds_name, data, seed, device):
    set_seed(seed)
    X = data["X"].to(device); A_hat = data["A_hat"].to(device); y = data["y"].to(device)

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
            if va > best_val: best_val = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    del optim, best_state; gc.collect()
    model.eval()

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx].clone()
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
    del X, A_hat, y, ctx; gc.collect(); torch.cuda.empty_cache()

    Z_sub = Z_star[idx].clone(); del Z_star
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7: break
            Z_sub = Z_new
        Z_sub = Z_new

    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    del J_z, J_A

    if not edge_list:
        del model, S, S_c; gc.collect()
        return None

    results = []
    for eps in EPS_VALUES:
        attack = optimal_structural_attack(S, A_sub, epsilon=eps)
        predicted_shift = eps * attack["sigma_1"]

        dA = torch.zeros_like(A_sub)
        U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
        weights = eps * Vh_c[0]
        for k, (i, j) in enumerate(edge_list):
            dA[i, j] = float(weights[k])
            dA[j, i] = float(weights[k])

        with torch.no_grad():
            ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
            Z = Z_sub.clone()
            for _ in range(200):
                Z_n = model.operator(Z, ctx_pert)
                if (Z_n - Z).norm() < 1e-7: break
                Z = Z_n
            actual_shift = float((Z_n - Z_sub).norm())

        tightness = actual_shift / max(predicted_shift, 1e-10)

        preds_clean = model.head(Z_sub).argmax(1)
        preds_pert = model.head(Z_n).argmax(1)
        flip_rate = float((preds_clean != preds_pert).float().mean())

        results.append({
            "dataset": ds_name, "seed": seed, "epsilon": eps,
            "tightness": tightness, "flip_rate": flip_rate,
        })

    # Weight-corrected tau (Proposition 3 verification)
    cont_scores = np.array([float(S_c[:, k].norm()) for k in range(len(edge_list))])
    edge_weights = np.array([float(A_sub[i, j]) for i, j in edge_list])
    weighted_scores = cont_scores * edge_weights

    disc_scores = np.array(greedy_structural_attack(model, Z_sub, ctx_sub))
    disc_damage = np.array([s for _, _, s in greedy_structural_attack(model, Z_sub, ctx_sub)])

    tau_raw, _ = kendalltau(cont_scores, disc_damage)
    tau_weighted, _ = kendalltau(weighted_scores, disc_damage)

    for r in results:
        r["tau_raw"] = tau_raw
        r["tau_weighted"] = tau_weighted

    del model, S, S_c, Z_sub, ctx_sub, A_sub; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_amazon import _load_amazon

    datasets = {
        "Pubmed": _load_planetoid("pubmed", Path("datasets/pubmed")),
        "Amazon Photo": _load_amazon(Path("datasets/amazon_photo")),
    }

    rows = []
    for ds_name, data in datasets.items():
        for seed_idx, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)})", end=" ", flush=True)
            try:
                r = run_tightness(ds_name, data, seed, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM", flush=True); gc.collect(); torch.cuda.empty_cache(); continue
                raise
            if r is None:
                print("SKIP", flush=True); continue
            rows.extend(r)
            t01 = [x["tightness"] for x in r if x["epsilon"] == 0.01][0]
            tw = r[0]["tau_weighted"]
            print(f"tight@0.01={t01:.3f} tau_w={tw:+.3f}", flush=True)

    csv_path = Path("results/tightness_expansion.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    import statistics
    print("\nTIGHTNESS DEGRADATION (Pubmed + Amazon Photo)")
    print(f"{'Dataset':<15} {'eps=.01':>10} {'eps=.05':>10} {'eps=.10':>10} {'eps=.20':>10} {'tau_raw':>10} {'tau_wt':>10}")
    for ds in ["Pubmed", "Amazon Photo"]:
        vals = {}
        for eps in EPS_VALUES:
            sub = [r["tightness"] for r in rows if r["dataset"] == ds and r["epsilon"] == eps]
            vals[eps] = f"{np.mean(sub):.2f}±{np.std(sub):.2f}" if sub else "N/A"
        tr = [r["tau_raw"] for r in rows if r["dataset"] == ds and r["epsilon"] == 0.01]
        tw = [r["tau_weighted"] for r in rows if r["dataset"] == ds and r["epsilon"] == 0.01]
        tr_s = f"{np.mean(tr):+.2f}" if tr else "N/A"
        tw_s = f"{np.mean(tw):+.2f}" if tw else "N/A"
        print(f"{ds:<15} {vals[0.01]:>10} {vals[0.05]:>10} {vals[0.10]:>10} {vals[0.20]:>10} {tr_s:>10} {tw_s:>10}")


if __name__ == "__main__":
    sys.exit(main() or 0)
