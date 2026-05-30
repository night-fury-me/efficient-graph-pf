"""X4 -- Fixed- vs recomputed-normalization edge deletion (validates prop:transfer's caveat).

prop:transfer models deletion as FIXED-normalization masking (zero the A_hat entry, D held
fixed) and states recompute-normalization adds an "O(d_i^{-1}) incident-edge rescaling,
NEGLIGIBLE OFF LOW-DEGREE NODES." X4 validates exactly that, across the full degree range:

  (1) OPERATOR correction (model-free, exact):  g_k = ||A_recompute(k) - A_fixed(k)||_F
      should scale as O(1/d_k), i.e. g_k * d_k ~ const  =>  negligible at high degree.
  (2) DAMAGE ranking (with model): Kendall tau between fixed- and recompute-normalization
      deletion damage; high among the high-degree edges that dominate the ranking (top-k),
      with O(1/d) reshuffling confined to the low-degree, low-damage tail.

The correction is measured on the FULL graph (real degrees, self-consistent normalization
A_hat = D^{-1/2}(A+I)D^{-1/2} rebuilt from the recovered binary edge set, so it is independent
of any per-loader normalization quirk). g_k uses only rows/cols i,j, so it is cheap for all
edges. The damage ranking is measured on a 50-node BFS subgraph (good single-edge SNR).

Outputs:
  results/exp_deletion_normalization.csv          (per-seed/dataset summary)
  results/exp_deletion_norm_peredge.csv           (per-edge g_k vs degree, first seed)

Usage: .venv/bin/python scripts/exp_deletion_normalization.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import extract_ego_subgraph
from iem.examples.ignn_amazon import _load_amazon
from iem.examples.ignn_cora import _load_cora
from exp_phase_transition import set_seed, train_ignn_kappa  # noqa: E402

KAPPA = 0.90
SUBGRAPH_NODES = 50
RECONVERGE_ITERS = 400
ALL_SEEDS = [42, 137, 271, 314, 1729]
DATASETS = [
    ("Cora",         lambda: _load_cora(Path("datasets/cora")),          3),
    ("Amazon Photo", lambda: _load_amazon(Path("datasets/amazon_photo")), 2),
]
MAX_OP_EDGES = 4000     # cap for the model-free operator-gap sweep


def build_self_consistent(A_in):
    """Rebuild A_hat = D^{-1/2}(A_raw+I)D^{-1/2} from the binary edge set of A_in (float64)."""
    A = A_in.double()
    N = A.shape[0]
    raw = (A.abs() > 1e-10).double()
    raw.fill_diagonal_(0.0)
    deg = raw.sum(1)                                            # node degree (no self-loop)
    cinv = (deg + 1.0).rsqrt()                                 # 1/sqrt(d+1)
    AI = raw + torch.eye(N, dtype=A.dtype, device=A.device)
    return cinv[:, None] * AI * cinv[None, :], raw, deg


def op_gap(A_hat, deg, i, j):
    """g_k = ||A_recompute(k) - A_fixed(k)||_F from rows i,j only (O(N), not O(N^2)).
    recompute-fixed = the renormalization rescaling of rows/cols i,j (the -w_k edge removal
    cancels). For symmetric A_hat, ||M||_F^2 = (A_ii(ri^2-1))^2 + (A_jj(rj^2-1))^2
      + 2(ri-1)^2 * sum_{b!=i,j} A_ib^2 + 2(rj-1)^2 * sum_{b!=i,j} A_jb^2."""
    di, dj = float(deg[i]), float(deg[j])
    if di < 1 or dj < 1:
        return None
    ri = ((di + 1.0) / di) ** 0.5
    rj = ((dj + 1.0) / dj) ** 0.5
    ai, aj = A_hat[i], A_hat[j]
    si = float((ai ** 2).sum() - ai[i] ** 2 - ai[j] ** 2)     # sum_{b!=i,j} A_ib^2
    sj = float((aj ** 2).sum() - aj[j] ** 2 - aj[i] ** 2)
    g2 = (float(ai[i]) * (ri ** 2 - 1.0)) ** 2 + (float(aj[j]) * (rj ** 2 - 1.0)) ** 2 \
        + 2.0 * (ri - 1.0) ** 2 * si + 2.0 * (rj - 1.0) ** 2 * sj
    return float(max(g2, 0.0) ** 0.5)


def op_gap_ref(A_hat, deg, i, j):
    """Full-matrix reference for op_gap (verification only)."""
    di, dj = float(deg[i]), float(deg[j])
    if di < 1 or dj < 1:
        return None
    ri = ((di + 1.0) / di) ** 0.5
    rj = ((dj + 1.0) / dj) ** 0.5
    rho = torch.ones(A_hat.shape[0], dtype=A_hat.dtype, device=A_hat.device)
    rho[i] = ri
    rho[j] = rj
    A_rec = A_hat * rho[:, None] * rho[None, :]
    A_rec[i, j] = 0.0
    A_rec[j, i] = 0.0
    A_fix = A_hat.clone()
    A_fix[i, j] = 0.0
    A_fix[j, i] = 0.0
    return float((A_rec - A_fix).norm())


def reconverge(model, A, X_proj, z0, iters=RECONVERGE_ITERS, tol=1e-9):
    ctx = {"A_hat": A.to(X_proj.dtype), "X_proj": X_proj}
    Z = z0.clone()
    with torch.no_grad():
        for _ in range(iters):
            Zn = model.operator(Z, ctx)
            if (Zn - Z).norm() < tol:
                break
            Z = Zn
    return Zn


def run_seed(name, data, device, seed, dump_peredge):
    set_seed(seed)
    model, Z_star, ctx, _ = train_ignn_kappa(data, device, seed, KAPPA)
    A_hat_full = data["A_hat"].to(device)

    # ---- (1) OPERATOR correction on the FULL graph, across the degree range ----
    A_sc, raw, deg = build_self_consistent(A_hat_full)         # float64, self-consistent
    iu, ju = torch.where(torch.triu(raw > 0.5, diagonal=1))
    edges_full = list(zip(iu.tolist(), ju.tolist()))
    if len(edges_full) > MAX_OP_EDGES:
        sel = np.linspace(0, len(edges_full) - 1, MAX_OP_EDGES).astype(int)
        edges_full = [edges_full[k] for k in sel]
    # verify analytic op_gap == full-matrix reference on the first valid edge
    for (i, j) in edges_full:
        if float(deg[i]) >= 1 and float(deg[j]) >= 1:
            g_a, g_r = op_gap(A_sc, deg, i, j), op_gap_ref(A_sc, deg, i, j)
            assert abs(g_a - g_r) < 1e-6 * (abs(g_r) + 1e-12), \
                f"op_gap analytic {g_a} != reference {g_r}"
            break

    gk, mind = [], []
    peredge = []
    for (i, j) in edges_full:
        g = op_gap(A_sc, deg, i, j)
        if g is None:
            continue
        d = min(float(deg[i]), float(deg[j]))
        gk.append(g)
        mind.append(d)
        if dump_peredge:
            peredge.append({"dataset": name, "i": i, "j": j, "min_deg": d, "g_k": g})
    gk, mind = np.array(gk), np.array(mind)
    # O(1/d): g_k * d should be ~ const; report its coefficient of variation + log-log slope
    gd = gk * mind
    cv = float(gd.std() / (gd.mean() + 1e-12))
    slope = float(np.polyfit(np.log(mind), np.log(gk + 1e-12), 1)[0])  # expect ~ -1
    # gap at high vs low degree (relative to the deleted edge weight w_k ~ 1/d)
    hi = mind >= np.quantile(mind, 0.9)
    lo = mind <= np.quantile(mind, 0.1)

    # ---- (2) DAMAGE ranking on a 50-node subgraph (single-edge SNR) ----
    idx = extract_ego_subgraph(A_hat_full, max_nodes=SUBGRAPH_NODES)
    A_raw_s = (A_hat_full[idx][:, idx].abs() > 1e-10).double()
    A_raw_s.fill_diagonal_(0.0)
    deg_s = A_raw_s.sum(1)
    Ns = A_raw_s.shape[0]
    AIs = A_raw_s + torch.eye(Ns, dtype=torch.float64, device=device)
    cinv_s = (deg_s + 1.0).rsqrt()
    A0 = (cinv_s[:, None] * AIs * cinv_s[None, :]).to(torch.float32)
    X_proj_s = ctx["X_proj"][idx].clone()
    Z0 = reconverge(model, A0, X_proj_s, Z_star[idx].clone())

    iu2, ju2 = torch.where(torch.triu(A_raw_s > 0.5, diagonal=1))
    sub_edges = list(zip(iu2.tolist(), ju2.tolist()))
    dfix, drec, mind_s = [], [], []
    for (i, j) in sub_edges:
        Af = A0.clone(); Af[i, j] = 0.0; Af[j, i] = 0.0
        dfix.append(float((reconverge(model, Af, X_proj_s, Z0) - Z0).norm()))
        raw2 = A_raw_s.clone(); raw2[i, j] = 0.0; raw2[j, i] = 0.0
        d2 = raw2.sum(1); cs = (d2 + 1.0).rsqrt()
        Ar = (cs[:, None] * (raw2 + torch.eye(Ns, dtype=torch.float64, device=device)) * cs[None, :]).to(torch.float32)
        drec.append(float((reconverge(model, Ar, X_proj_s, Z0) - Z0).norm()))
        mind_s.append(min(float(deg_s[i]), float(deg_s[j])))
    dfix, drec, mind_s = np.array(dfix), np.array(drec), np.array(mind_s)
    tau, _ = kendalltau(dfix, drec)
    # top-k overlap (the edges AEGIS actually flags)
    def patk(a, b, k):
        k = min(k, len(a))
        return len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:])) / k if k else float("nan")
    p10, p20 = patk(dfix, drec, 10), patk(dfix, drec, 20)

    return {"tau": float(tau), "p_at_10": p10, "p_at_20": p20,
            "op_slope_loglog": slope, "op_gd_cv": cv,
            "n_full_edges": len(gk), "n_sub_edges": len(sub_edges),
            "gk_lo_deg": float(gk[lo].mean()), "deg_lo": float(mind[lo].mean()),
            "gk_hi_deg": float(gk[hi].mean()), "deg_hi": float(mind[hi].mean()),
            "peredge": peredge}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    rows, all_peredge, t0 = [], [], time.time()

    for name, loader, n_seeds in DATASETS:
        print(f"\n{'='*64}\n{name}\n{'='*64}", flush=True)
        data = loader()
        print(f"  N={data['N']}, feat={data['n_features']}, classes={data['n_classes']}", flush=True)
        for si, seed in enumerate(ALL_SEEDS[:n_seeds]):
            r = run_seed(name, data, device, seed, dump_peredge=(si == 0))
            all_peredge.extend(r.pop("peredge"))
            print(f"  [seed {seed}] OPERATOR g_k~O(1/d): loglog_slope={r['op_slope_loglog']:.3f} "
                  f"(expect ~-1)  g_k*d CV={r['op_gd_cv']:.3f}  | "
                  f"g_k(low deg~{r['deg_lo']:.0f})={r['gk_lo_deg']:.4f}  "
                  f"g_k(high deg~{r['deg_hi']:.0f})={r['gk_hi_deg']:.4f}", flush=True)
            print(f"  [seed {seed}] DAMAGE ranking: tau={r['tau']:.4f}  "
                  f"P@10={r['p_at_10']:.2f}  P@20={r['p_at_20']:.2f}  "
                  f"({r['n_sub_edges']} sub-edges, {r['n_full_edges']} full-edges)", flush=True)
            rows.append({"dataset": name, "seed": seed, **r})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    out = Path("results/exp_deletion_normalization.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["dataset", "seed", "tau", "p_at_10", "p_at_20",
                                           "op_slope_loglog", "op_gd_cv", "n_full_edges",
                                           "n_sub_edges", "gk_lo_deg", "deg_lo", "gk_hi_deg", "deg_hi"])
        wr.writeheader()
        wr.writerows(rows)
    with open(Path("results/exp_deletion_norm_peredge.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["dataset", "i", "j", "min_deg", "g_k"])
        wr.writeheader()
        wr.writerows(all_peredge)
    print(f"Saved {out} and per-edge CSV")

    print("\n" + "=" * 64)
    for name, _, _ in DATASETS:
        rr = [r for r in rows if r["dataset"] == name]
        if not rr:
            continue
        print(f"{name}:  operator g_k loglog-slope={np.mean([r['op_slope_loglog'] for r in rr]):.3f} "
              f"(O(1/d) => -1)   damage tau={np.mean([r['tau'] for r in rr]):.3f}   "
              f"P@20={np.mean([r['p_at_20'] for r in rr]):.2f}")
    print("\nClaim: g_k ~ O(1/d) (slope ~ -1) => recompute correction negligible at high degree;")
    print("top-k agreement high => the edges AEGIS flags are ranked the same either way.")


if __name__ == "__main__":
    sys.exit(main() or 0)
