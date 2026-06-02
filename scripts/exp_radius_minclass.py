"""X5 -- Min-over-classes radius recompute (validates the corrected prop:radius / T3).

The implementation (`per_node_robust_radius`) and Algorithm 1 use the OLD runner-up surrogate
    r_hat_v = m_v / (||W_{y_v} - W_{c*}||_2 * ||S_v||_2)      (runner-up c* only, PRODUCT norm)
while the corrected prop:radius is the min-over-classes COMPOSED-norm form
    r_v = min_{c != y_v} m_v^{(c)} / ||(W_{y_v} - W_c) S_v||_2 .

X5 recomputes r_v both ways (prediction-based, theory-correct), reports how the headline
numbers move, and re-validates the breach guarantee "every breached node satisfies eps > r_v"
(false-safe rate) under the corrected radius -- the safety claim must still hold.

Two effects fight: composed <= product norm makes each per-class radius LARGER, but the min
over ALL competitors (not just the runner-up) can be SMALLER -- the theory says the surrogate
"can be optimistic." X5 measures the net.

Outputs: results/exp_radius_minclass.csv
Usage: .venv/bin/python scripts/exp_radius_minclass.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_citeseer_pubmed import _load_planetoid
from iem.examples.ignn_cora import _load_cora
from exp_phase_transition import set_seed, train_ignn_kappa  # noqa: E402

KAPPA = 0.90
SUBGRAPH_NODES = 50
EPSILONS = [0.05, 0.10, 0.20]
ALL_SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: ALL_SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
DATASETS = [
    ("Cora",     lambda: _load_cora(Path("datasets/cora")),                       10),
    ("Citeseer", lambda: _load_planetoid("citeseer", Path("datasets/citeseer")),  10),
]


def both_radii(S, z_star, logits, head):
    """Return (r_surrogate, r_minclass) per node, prediction-based (theory-correct).
    r_surrogate: m_v / (||W_pred - W_c*|| * ||S_v||)  (runner-up c*, product norm)
    r_minclass : min_{c != pred} (f_pred - f_c) / ||(W_pred - W_c) S_v||_2  (composed norm)."""
    N, d = z_star.shape
    C = logits.shape[1]
    W = head.weight.detach()                                   # (C, d)
    L = logits.detach()
    pred = L.argmax(1)
    r_sur = torch.zeros(N, device=W.device)
    r_min = torch.zeros(N, device=W.device)
    for v in range(N):
        p = int(pred[v])
        Sv = S[v * d:(v + 1) * d]                              # (d, |E|)
        Snorm = float(Sv.norm())
        margins = L[v, p] - L[v]                               # (C,) ; margins[c] = f_pred - f_c
        margins[p] = float("inf")                              # exclude own class
        cstar = int(torch.argmin(margins))                    # runner-up = smallest positive margin
        # --- surrogate (runner-up, product norm) ---
        wdiff_star = W[p] - W[cstar]
        r_sur[v] = float(margins[cstar]) / (float(wdiff_star.norm()) * Snorm + 1e-12)
        # --- min-over-classes (composed norm) ---
        best = float("inf")
        for c in range(C):
            if c == p:
                continue
            m_c = float(margins[c])
            comp = (W[p] - W[c]) @ Sv                          # (|E|,)  composed norm
            r_c = m_c / (float(comp.norm()) + 1e-12)
            best = min(best, r_c)
        r_min[v] = max(best, 0.0)
    return r_sur.cpu(), r_min.cpu()


def run_seed(name, data, device, seed):
    set_seed(seed)
    model, Z_star, ctx, _ = train_ignn_kappa(data, device, seed, KAPPA)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    idx = extract_ego_subgraph(A_hat, max_nodes=SUBGRAPH_NODES)
    A_sub = A_hat[idx][:, idx].clone()
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
    labels_sub = y[idx]

    Z = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(300):
            Zn = model.operator(Z, ctx_sub)
            if (Zn - Z).norm() < 1e-8:
                break
            Z = Zn
    Z_sub = Zn
    with torch.no_grad():
        logits = model.head(Z_sub)
    preds_clean = logits.argmax(1)

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)
    if rho >= 1.0:
        return None

    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
                                      J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0 or not edge_list:
        return None

    r_sur, r_min = both_radii(S, Z_sub, logits, model.head)
    r_sur, r_min = r_sur.to(device), r_min.to(device)

    # breach false-safe: apply S_c-optimal perturbation, count breaches with eps < r_v
    Vh = torch.linalg.svd(S_c, full_matrices=False)[2]
    out = {"r_sur_mean": float(r_sur.mean()), "r_sur_med": float(r_sur.median()),
           "r_min_mean": float(r_min.mean()), "r_min_med": float(r_min.median()),
           "n_nodes": int(len(idx))}
    for eps in EPSILONS:
        dA = torch.zeros_like(A_sub)
        w = eps * Vh[0]
        for k, (i, j) in enumerate(edge_list):
            dA[i, j] = float(w[k]); dA[j, i] = float(w[k])
        ctx_p = {**ctx_sub, "A_hat": A_sub + dA}
        Zp = Z_sub.clone()
        with torch.no_grad():
            for _ in range(300):
                Zn = model.operator(Zp, ctx_p)
                if torch.isnan(Zn).any() or Zn.norm() > 1e6:
                    Zn = Zp; break
                if (Zn - Zp).norm() < 1e-8:
                    break
                Zp = Zn
            preds_p = model.head(Zn).argmax(1)
        breached = (preds_p != preds_clean)
        nb = int(breached.sum())
        # false-safe: breached but eps < r_v (radius said "safe" yet it flipped)
        fs_sur = int(((breached) & (eps < r_sur)).sum())
        fs_min = int(((breached) & (eps < r_min)).sum())
        out[f"breach@{eps}"] = nb
        out[f"falsesafe_sur@{eps}"] = fs_sur
        out[f"falsesafe_min@{eps}"] = fs_min
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    rows, t0 = [], time.time()
    for name, loader, n_seeds in DATASETS:
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
        data = loader()
        for seed in ALL_SEEDS[:n_seeds]:
            r = run_seed(name, data, device, seed)
            if r is None:
                print(f"  [seed {seed}] skipped (rho>=1 or no edges)", flush=True)
                continue
            print(f"  [seed {seed}] r_v med: surrogate={r['r_sur_med']:.4f} -> "
                  f"minclass={r['r_min_med']:.4f}  | breaches/false-safe(sur,min) "
                  + " ".join(f"e{e}:{r[f'breach@{e}']}/{r[f'falsesafe_sur@{e}']},{r[f'falsesafe_min@{e}']}"
                             for e in EPSILONS), flush=True)
            rows.append({"dataset": name, "seed": seed, **r})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    out = Path("results/exp_radius_minclass.csv")
    out.parent.mkdir(exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"Saved {out}")

    print("\n" + "=" * 60)
    for name, _, _ in DATASETS:
        rr = [r for r in rows if r["dataset"] == name]
        if not rr:
            continue
        sur = np.mean([r["r_sur_med"] for r in rr]); mn = np.mean([r["r_min_med"] for r in rr])
        tb = sum(r[f"breach@{e}"] for r in rr for e in EPSILONS)
        fs_s = sum(r[f"falsesafe_sur@{e}"] for r in rr for e in EPSILONS)
        fs_m = sum(r[f"falsesafe_min@{e}"] for r in rr for e in EPSILONS)
        print(f"{name}: median r_v surrogate={sur:.4f} -> minclass={mn:.4f} "
              f"({100*(mn-sur)/sur:+.0f}%);  total breaches={tb}  "
              f"false-safe surrogate={fs_s} minclass={fs_m}")
    print("\nBreach guarantee 'every breached node has eps > r_v' holds iff false-safe == 0.")


if __name__ == "__main__":
    sys.exit(main() or 0)
