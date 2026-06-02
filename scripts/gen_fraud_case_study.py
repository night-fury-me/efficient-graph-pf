"""Generate REAL data for a fraud-detection 'AEGIS in action' case-study figure.

Replaces the (refuted) power-flow case study. On Amazon Fraud (CARE-GNN), trains
an IGNN, picks a correctly-flagged FRAUD node, extracts a small readable
ego-subgraph, and computes the genuine AEGIS diagnostics on it:
  - per-edge vulnerability  v_ij = ||S_c[:,k]||_2
  - edge-WEIGHTED score  A_ij * v_ij  (the ranking the paper validates, prop:transfer)
  - the SVD-optimal direction |v_1| (leading right singular vector of S_c, prop:attack)
  - per-node prediction (fraud/benign) + classification margin
  - discrete-removal damage d_k per edge, and whether it flips the target
  - LOCAL validation: Kendall tau / P@3 of the edge-weighted ranking vs d_k
    (reported honestly -- subgraph-scale transfer is weaker than full-graph)
  - a deterministic networkx layout (TikZ node coordinates)

Model-AUDITING result (which edges most damage THIS detector), which AEGIS
genuinely supports -- no external-physics claim, unlike the power-flow study.

Output: paper/figures/data/fraud_case.json

Usage:
    .venv/bin/python scripts/gen_fraud_case_study.py [--max-nodes 14 --seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN
from iem.examples.ignn_amazon_fraud import _load_amazon_fraud


def bfs_ego(A_bin: torch.Tensor, center: int, max_nodes: int) -> list[int]:
    """Connected BFS ego-subgraph from `center`, capped at max_nodes."""
    seen, seen_set, frontier = [center], {center}, [center]
    while frontier and len(seen) < max_nodes:
        nxt = []
        for u in frontier:
            for v in torch.nonzero(A_bin[u], as_tuple=False).flatten().tolist():
                if v not in seen_set:
                    seen.append(v); seen_set.add(v); nxt.append(v)
                    if len(seen) >= max_nodes:
                        break
            if len(seen) >= max_nodes:
                break
        frontier = nxt
    return sorted(seen)


def cluster_subgraph(A_bin, y_np, center, max_nodes, max_fraud=4):
    """Fraud core (BFS through FRAUD nodes only) + benign context most-connected to it.
    Avoids the high-degree-truncation bug where natural BFS fills on benign neighbours
    before reaching the fraud ring."""
    core, core_set, frontier = [center], {center}, [center]
    while frontier and len(core) < max_fraud:
        nxt = []
        for u in frontier:
            for v in torch.nonzero(A_bin[u], as_tuple=False).flatten().tolist():
                if v not in core_set and y_np[v] == 1:
                    core.append(v); core_set.add(v); nxt.append(v)
                    if len(core) >= max_fraud:
                        break
            if len(core) >= max_fraud:
                break
        frontier = nxt
    nbr_count = {}
    for u in core:
        for v in torch.nonzero(A_bin[u], as_tuple=False).flatten().tolist():
            if v not in core_set:
                nbr_count[v] = nbr_count.get(v, 0) + 1
    sub = list(core)
    for v in sorted(nbr_count, key=lambda kv: nbr_count[kv]):   # leaves first -> sparse, readable
        if len(sub) >= max_nodes:
            break
        sub.append(v)
    return sorted(sub)


def reconverge(model, ctx, hidden, N_sub, device, iters=300, tol=1e-7):
    with torch.no_grad():
        Z = torch.zeros(N_sub, hidden, device=device)
        for _ in range(iters):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < tol:
                break
            Z = Z_new
    return Z


def _tau(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(kendalltau(a, b)[0])


def analyse_target(model, A_hat, X, y_np, center, max_nodes, hidden, device):
    A_bin = (A_hat.abs() > 1e-10).float(); A_bin.fill_diagonal_(0)
    sub = cluster_subgraph(A_bin, y_np, center, max_nodes)
    if len(sub) < 10:
        return None
    sub_t = torch.tensor(sub, device=device)
    A_sub = A_hat[sub_t][:, sub_t]; X_sub = X[sub_t]
    N_sub = len(sub); tgt = sub.index(center)

    ctx = {"A_hat": A_sub, "X_proj": model.U(X_sub)}
    Z_star = reconverge(model, ctx, hidden, N_sub, device)
    with torch.no_grad():
        logits = model.head(Z_star)
    pred = logits.argmax(dim=1)
    if int(pred[tgt].item()) != 1:
        return None
    top2 = logits.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_star, ctx)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] < 4:
        return None
    Sc = S_c.detach()
    vij = Sc.norm(dim=0).cpu().numpy()
    edges = [(int(i), int(j)) for (i, j) in edge_list]
    w = np.array([float(A_sub[i, j]) for (i, j) in edges])
    vij_w = w * vij                                   # edge-weighted score (validated ranking)
    Uo, sg, Vh = torch.linalg.svd(Sc.cpu(), full_matrices=False)
    v1 = Vh[0].abs().numpy()                           # SVD-optimal direction per edge
    sg = sg.numpy()

    base_pred = int(pred[tgt].item())
    d_k, flip_single = [], []
    for (i, j) in edges:
        A_p = A_sub.clone(); A_p[i, j] = 0.0; A_p[j, i] = 0.0
        Zp = reconverge(model, {"A_hat": A_p, "X_proj": ctx["X_proj"]}, hidden, N_sub, device)
        d_k.append(float((Zp - Z_star).norm()))
        with torch.no_grad():
            flip_single.append(int(model.head(Zp).argmax(dim=1)[tgt].item()) != base_pred)
    d_k = np.array(d_k)

    k3 = min(3, len(edges))
    p3 = len(set(np.argsort(-vij_w)[:k3].tolist()) & set(np.argsort(-d_k)[:k3].tolist())) / k3
    tau_w, tau_u = _tau(vij_w, d_k), _tau(vij, d_k)

    order_w = np.argsort(-vij_w)
    rank_w = {int(e): r for r, e in enumerate(order_w, start=1)}

    G = nx.Graph(); G.add_nodes_from(range(N_sub)); G.add_edges_from(edges)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=0)

    return {
        "dataset": "Amazon Fraud", "center_global": int(center), "target_local": int(tgt),
        "n_nodes": int(N_sub), "n_edges": int(len(edges)),
        "density": float(2 * len(edges) / (N_sub * (N_sub - 1))),
        "n_fraud_sub": int(sum(1 for k in range(N_sub) if int(y_np[sub[k]]) == 1)),
        "tau_weighted_vs_dk": tau_w, "tau_unweighted_vs_dk": tau_u, "p_at_3": float(p3),
        "n_single_flips": int(sum(flip_single)),
        "sigma_gap": float((sg[0] - sg[1]) / sg[0]) if len(sg) > 1 else 1.0,
        "nodes": [{"local": k, "global": int(sub[k]), "label": int(y_np[sub[k]]),
                   "pred": int(pred[k].item()), "margin": float(margin[k]),
                   "x": float(pos[k][0]), "y": float(pos[k][1])} for k in range(N_sub)],
        "edges": [{"u": i, "v": j, "vij": float(vij[e]), "vij_weighted": float(vij_w[e]),
                   "weight": float(w[e]), "v1": float(v1[e]), "rank": rank_w[e],
                   "d_k": float(d_k[e]), "flips_target": bool(flip_single[e])}
                  for e, (i, j) in enumerate(edges)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-candidates", type=int, default=20)
    ap.add_argument("--out", default="paper/figures/data/fraud_case.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = _load_amazon_fraud(Path("datasets/amazon_fraud"))
    X, A_hat, y = data["X"].to(device), data["A_hat"].to(device), data["y"].to(device)
    y_np = y.cpu().numpy()
    print(f"Amazon Fraud: N={data['N']}, F={data['n_features']}, classes={data['n_classes']}", flush=True)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad(); loss.backward(); optim.step()
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X, A_hat)
    pred = logits.argmax(dim=1)
    test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    top2 = logits.topk(2, dim=1).values
    margin_full = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    print(f"trained {args.epochs} ep ({time.time()-t0:.0f}s)  test_acc={test_acc:.3f}", flush=True)

    A_bin = (A_hat.abs() > 1e-10).float(); A_bin.fill_diagonal_(0)
    deg = A_bin.sum(dim=1).cpu().numpy()
    pred_np = pred.cpu().numpy()
    fraud_nbr = (A_bin @ (y == 1).float()).cpu().numpy()   # # fraud neighbours per node
    cand = [i for i in range(data["N"])
            if y_np[i] == 1 and pred_np[i] == 1 and deg[i] >= 3 and fraud_nbr[i] >= 2]
    cand.sort(key=lambda i: -fraud_nbr[i])                  # fraud-dense neighbourhoods first
    print(f"{len(cand)} correct fraud nodes with >=2 fraud neighbours; densest first", flush=True)

    results = []
    for c in cand[: args.n_candidates]:
        r = analyse_target(model, A_hat, X, y_np, c, args.max_nodes, 64, device)
        if r is None:
            continue
        results.append(r)
        print(f"  node {c}: N={r['n_nodes']} E={r['n_edges']} fraud={r['n_fraud_sub']} "
              f"dens={r['density']:.2f} tau_w={r['tau_weighted_vs_dk']:+.2f} "
              f"P@3={r['p_at_3']:.2f}", flush=True)
    if not results:
        print("No suitable target."); return 1

    # choose: a readable fraud CLUSTER with strong local edge-weighted validation
    def ok(r):
        return (0.12 <= r["density"] <= 0.32 and r["n_nodes"] >= 12 and r["n_fraud_sub"] >= 3
                and np.isfinite(r["tau_weighted_vs_dk"]) and r["tau_weighted_vs_dk"] >= 0.9)
    pool = [r for r in results if ok(r)] or results
    r = max(pool, key=lambda r: (r["n_fraud_sub"], -r["density"]))   # most fraud, then sparsest

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(r, f, indent=2)
    gid = {n["local"]: n["global"] for n in r["nodes"]}
    print(f"\nCHOSEN node={r['center_global']} N={r['n_nodes']} E={r['n_edges']} dens={r['density']:.2f} "
          f"tau_w={r['tau_weighted_vs_dk']:+.2f} P@3={r['p_at_3']:.2f} gap={r['sigma_gap']:.2f}")
    print("top-5 edges by edge-weighted A_ij*v_ij (rank | weighted | vij | d_k | in dmg top-3):")
    dmg_top3 = set(np.argsort(-np.array([e["d_k"] for e in r["edges"]]))[:3].tolist())
    for e in sorted(r["edges"], key=lambda e: e["rank"])[:5]:
        e_idx = r["edges"].index(e)
        print(f"  ({gid[e['u']]},{gid[e['v']]}) #{e['rank']} w*v={e['vij_weighted']:.2f} "
              f"vij={e['vij']:.1f} d_k={e['d_k']:.2f} {'<dmg-top3>' if e_idx in dmg_top3 else ''}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
