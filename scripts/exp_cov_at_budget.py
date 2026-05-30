"""Cov%@0.05: fraction of subgraph nodes certified first-order-safe at budget rho=0.05.

Under the corrected min-over-classes radius (prop:radius), 'fraction with POSITIVE r_v' is
trivially ~100% (every predicted node has a positive radius), so the meaningful coverage metric
is at a fixed budget:  Cov%@rho = mean_v 1[r_v > rho],  rho = 0.05 (matching smoothing sigma=0.05
and the breach experiment's smallest budget). Larger corrected radii => higher coverage.

5 datasets x 10 seeds, 50-node BFS subgraph, standard IGNN (full-graph trained).
Outputs: results/exp_cov_at_budget.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    extract_ego_subgraph,
    per_node_robust_radius,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_amazon import _load_amazon
from iem.examples.ignn_citeseer_pubmed import _load_planetoid
from iem.examples.ignn_cora import IGNN, _load_cora
from iem.examples.ignn_wikics import _load_wikics

RHO = 0.05
SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
DATASETS = [
    ("Cora",     lambda: _load_cora(Path("datasets/cora"))),
    ("Citeseer", lambda: _load_planetoid("citeseer", Path("datasets/citeseer"))),
    ("Pubmed",   lambda: _load_planetoid("pubmed", Path("datasets/pubmed"))),
    ("Amazon",   lambda: _load_amazon(Path("datasets/amazon_photo"))),
    ("WikiCS",   lambda: _load_wikics(Path("datasets/wikics"))),
]


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train(model, X, A, y, mask, epochs=120):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _, _ = model(X, A)
        F_func.cross_entropy(logits[mask], y[mask]).backward()
        opt.step()
    model.eval()
    return model


def run(name, data, device, seed):
    set_seed(seed)
    X, A, y = data["X"].to(device), data["A_hat"].to(device), data["y"].to(device)
    model = IGNN(data["n_features"], 64, data["n_classes"]).to(device)
    train(model, X, A, y, data["train_mask"].to(device))
    idx = extract_ego_subgraph(A, max_nodes=50)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A)
    A_sub = A[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(300):
            Zn = model.operator(Z, ctx_sub)
            if (Zn - Z).norm() < 1e-8:
                break
            Z = Zn
        logits = model.head(Zn)
    rho = spectral_radius(lambda z: model.operator(z.reshape(Zn.shape), ctx_sub).reshape(-1), Zn)
    if rho >= 1.0:
        return None
    Jz, JA, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Zn, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Zn, ctx_sub, J_z=Jz, J_A=JA)
    r = per_node_robust_radius(S, Zn, logits, y[idx], rho, model.head)["radii"]
    return {"dataset": name, "seed": seed, "cov05": float((r > RHO).float().mean()),
            "median_r": float(r.median()), "n": int(len(idx))}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  rho={RHO}", flush=True)
    rows, t0 = [], time.time()
    for name, loader in DATASETS:
        data = loader()
        print(f"\n{name} (N={data['N']}):", flush=True)
        for seed in SEEDS:
            r = run(name, data, device, seed)
            if r:
                rows.append(r)
                print(f"  seed {seed}: cov@0.05={r['cov05']:.1%}  med_r={r['median_r']:.3f}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    out = Path("results/exp_cov_at_budget.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "cov05", "median_r", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}\n" + "=" * 50)
    for name, _ in DATASETS:
        c = [r["cov05"] for r in rows if r["dataset"] == name]
        m = [r["median_r"] for r in rows if r["dataset"] == name]
        if c:
            print(f"{name:10s} Cov@0.05 = {100*np.mean(c):.0f} +/- {100*np.std(c):.0f}   "
                  f"(median r_v {np.mean(m):.3f}, {len(c)} seeds)")


if __name__ == "__main__":
    sys.exit(main() or 0)
