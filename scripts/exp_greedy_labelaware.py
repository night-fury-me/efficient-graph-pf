"""Baseline remediation item 2 — genuine LABEL-AWARE Greedy oracle for fig:greedy_topk.

The original `exp_greedy_topk_attack.py` greedily maximizes the LABEL-FREE equilibrium
shift ||Z_pert - Z_clean|| yet the figure caption calls it "label-aware Greedy ...
AEGIS recovers 54-67% ... with no label access" (baseline audit -> greedy.md). Both the
oracle and AEGIS are label-free there, so that contrast is fictional.

This script builds a TRUE label-aware oracle: a greedy that at each step removes the edge
maximizing the classification loss CE(head(Z_pert), y_true) on the subgraph's TRUE labels.
AEGIS uses only its label-free S_c column-norm ranking. We then score PREDICTION damage
(accuracy drop / prediction flips / CE increase) so that AEGIS recovering the oracle's
damage genuinely earns "no label access". The shift-oracle (Greedy-shift) and the shift
metric are also reported for continuity with the original 54-67% number.

Identical harness to the original (IGNN c=0.9/dropout=0.5, 50-node BFS ego-subgraph,
10 preferred seeds, k=1..10) so numbers are comparable.

Output: results/greedy_labelaware.csv

Usage:
    .venv/bin/python scripts/exp_greedy_labelaware.py            # all 10 seeds
    AEGIS_SEEDS=42 .venv/bin/python scripts/exp_greedy_labelaware.py   # smoke
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
_s = os.environ.get("AEGIS_SEEDS")
if _s:
    SEEDS = [int(x) for x in _s.split(",") if x.strip()]
MAX_K = 10
N_RANDOM_SHUFFLES = 5
DATASET_NAMES = ["Cora", "Citeseer", "WikiCS"]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconverge(model, Z_init, ctx, max_iter=200):
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return Z_new


def load_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics
    print("Loading datasets...", flush=True)
    return {
        "Cora": _load_cora(Path("datasets/cora")),
        "Citeseer": _load_planetoid("citeseer", Path("datasets/citeseer")),
        "WikiCS": _load_wikics(Path("datasets/wikics")),
    }


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
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = model(X, A_hat)
                va = float((lv.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def subgraph_rankings(model, data, device, seed):
    """Return (Z_sub, ctx_sub, edge_list, aegis_order, degree_order, y_sub, logits_clean)."""
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)
    y_sub = y[idx]
    with torch.no_grad():
        logits_clean = model.head(Z_sub)

    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    col_norms = torch.stack([S_c[:, k].norm() for k in range(S_c.shape[1])])
    aegis_order = col_norms.argsort(descending=True).tolist()
    deg = (A_sub.abs() > 1e-10).float().sum(dim=1)
    deg_scores = torch.tensor([max(float(deg[i]), float(deg[j])) for i, j in edge_list])
    degree_order = deg_scores.argsort(descending=True).tolist()
    return Z_sub, ctx_sub, edge_list, aegis_order, degree_order, y_sub, logits_clean


def measure_cumulative(model, Z_clean, ctx_sub, edge_list, order, y_sub, logits_clean, max_k):
    """Cumulative metrics along a removal order: shift, accuracy, flip-fraction, CE."""
    A_cur = ctx_sub["A_hat"].clone()
    pred_clean = logits_clean.argmax(1)
    out = []
    for step in range(min(max_k, len(order))):
        i, j = edge_list[order[step]]
        A_cur = A_cur.clone()
        A_cur[i, j] = 0.0
        A_cur[j, i] = 0.0
        Zp = reconverge(model, Z_clean, {**ctx_sub, "A_hat": A_cur})
        with torch.no_grad():
            lp = model.head(Zp)
            shift = float((Zp - Z_clean).norm())
            acc = float((lp.argmax(1) == y_sub).float().mean())
            flips = float((lp.argmax(1) != pred_clean).float().mean())
            ce = float(F_func.cross_entropy(lp, y_sub).item())
        out.append({"shift": shift, "acc": acc, "flips": flips, "ce": ce})
    return out


def greedy_order(model, Z_clean, ctx_sub, edge_list, max_k, objective, y_sub=None):
    """Sequential greedy. objective in {'shift','ce'}; 'ce' is the LABEL-AWARE oracle."""
    A_cur = ctx_sub["A_hat"].clone()
    remaining = list(range(len(edge_list)))
    order = []
    for _ in range(min(max_k, len(remaining))):
        best_score, best = -1.0, -1
        for ei in remaining:
            i, j = edge_list[ei]
            At = A_cur.clone()
            At[i, j] = 0.0
            At[j, i] = 0.0
            Zt = reconverge(model, Z_clean, {**ctx_sub, "A_hat": At})
            with torch.no_grad():
                if objective == "ce":
                    score = float(F_func.cross_entropy(model.head(Zt), y_sub).item())
                else:  # shift
                    score = float((Zt - Z_clean).norm())
            if score > best_score:
                best_score, best = score, ei
        order.append(best)
        remaining.remove(best)
        i, j = edge_list[best]
        A_cur = A_cur.clone()
        A_cur[i, j] = 0.0
        A_cur[j, i] = 0.0
    return order


def run_single(ds_name, data, seed, device):
    model = train_ignn(data, device, seed)
    res = subgraph_rankings(model, data, device, seed)
    if res is None:
        return None
    Z_sub, ctx_sub, edge_list, aegis_order, degree_order, y_sub, logits_clean = res
    k = min(MAX_K, len(edge_list))

    # Orders
    ce_greedy = greedy_order(model, Z_sub, ctx_sub, edge_list, k, "ce", y_sub)   # label-aware oracle
    shift_greedy = greedy_order(model, Z_sub, ctx_sub, edge_list, k, "shift")    # continuity oracle

    orders = {
        "AEGIS": aegis_order,
        "Greedy-CE": ce_greedy,
        "Greedy-shift": shift_greedy,
        "Degree": degree_order,
    }
    clean_acc = float((logits_clean.argmax(1) == y_sub).float().mean())
    rows = []
    for name, order in orders.items():
        m = measure_cumulative(model, Z_sub, ctx_sub, edge_list, order, y_sub, logits_clean, k)
        for ki, d in enumerate(m):
            rows.append({"dataset": ds_name, "seed": seed, "method": name,
                         "k": ki + 1, "clean_acc": clean_acc, **d})

    # Random: average metrics over N_RANDOM_SHUFFLES
    rand_runs = []
    for _ in range(N_RANDOM_SHUFFLES):
        ro = list(range(len(edge_list)))
        random.shuffle(ro)
        rand_runs.append(measure_cumulative(model, Z_sub, ctx_sub, edge_list, ro, y_sub, logits_clean, k))
    for ki in range(k):
        avg = {key: float(np.mean([rand_runs[s][ki][key] for s in range(N_RANDOM_SHUFFLES)]))
               for key in ("shift", "acc", "flips", "ce")}
        rows.append({"dataset": ds_name, "seed": seed, "method": "Random",
                     "k": ki + 1, "clean_acc": clean_acc, **avg})

    return rows, clean_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t0 = time.time()
    datasets = load_datasets()
    Path("results").mkdir(exist_ok=True)
    all_rows = []
    for ds_name in DATASET_NAMES:
        data = datasets[ds_name]
        for si, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({si+1}/{len(SEEDS)})", end="", flush=True)
            try:
                out = run_single(ds_name, data, seed, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(" OOM SKIP", flush=True)
                    continue
                raise
            if out is None:
                print(" SKIP", flush=True)
                continue
            rows, clean_acc = out
            all_rows.extend(rows)
            ce5 = [r for r in rows if r["method"] == "Greedy-CE" and r["k"] == 5]
            ae5 = [r for r in rows if r["method"] == "AEGIS" and r["k"] == 5]
            if ce5 and ae5:
                print(f"  clean_acc={clean_acc:.2f}  @k=5 accdrop: "
                      f"CE-oracle={clean_acc-ce5[0]['acc']:.3f} AEGIS={clean_acc-ae5[0]['acc']:.3f}",
                      flush=True)

    csv_path = Path("results/greedy_labelaware.csv")
    fields = ["dataset", "seed", "method", "k", "clean_acc", "shift", "acc", "flips", "ce"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV: {csv_path}  ({time.time()-t0:.0f}s)\n", flush=True)

    # ---- Summary: AEGIS (label-free) vs label-aware CE-oracle on prediction damage ----
    def vals(ds, method, k, key):
        v = [r[key] for r in all_rows if r["dataset"] == ds and r["method"] == method and r["k"] == k]
        return float(np.mean(v)) if v else float("nan")

    def clean_of(ds):
        v = [r["clean_acc"] for r in all_rows if r["dataset"] == ds]
        return float(np.mean(v)) if v else float("nan")

    print("=" * 96)
    print("LABEL-AWARE RECOVERY  (AEGIS uses NO labels; Greedy-CE is the label-aware oracle)")
    print("=" * 96)
    print(f"{'Dataset':<10}{'k':>3} | {'clean_acc':>10}{'oracle_acc':>11}{'aegis_acc':>10} | "
          f"{'accdrop_recov%':>15}{'flip_recov%':>13}")
    for ds in DATASET_NAMES:
        ca = clean_of(ds)
        for k in (5, 10):
            oa, aa = vals(ds, "Greedy-CE", k, "acc"), vals(ds, "AEGIS", k, "acc")
            of, af = vals(ds, "Greedy-CE", k, "flips"), vals(ds, "AEGIS", k, "flips")
            d_or, d_ae = ca - oa, ca - aa
            rec_acc = (d_ae / d_or * 100) if d_or > 1e-9 else float("nan")
            rec_fl = (af / of * 100) if of > 1e-9 else float("nan")
            print(f"{ds:<10}{k:>3} | {ca:>10.3f}{oa:>11.3f}{aa:>10.3f} | {rec_acc:>14.1f}%{rec_fl:>12.1f}%")
    print("-" * 96)
    print("Continuity: AEGIS / Greedy-shift on the SHIFT metric (original 54-67% Cora claim)")
    for ds in DATASET_NAMES:
        for k in (5, 10):
            ae, gs = vals(ds, "AEGIS", k, "shift"), vals(ds, "Greedy-shift", k, "shift")
            r = ae / gs if gs > 1e-9 else float("nan")
            print(f"  {ds:<10} k={k:<3} AEGIS/Greedy-shift = {r:.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
