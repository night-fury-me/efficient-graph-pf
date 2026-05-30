"""X4 bulletproofing -- the high-vulnerability edges agree under fixed vs recompute deletion.

Defense layer 3: on the FULL graph (real degrees), the highest-damage edges are high-degree,
where the recompute-normalization correction is negligible (prop:transfer's O(1/d)). We show,
with float reconverge over edges stratified across the degree range:
  (1) corr(damage, degree) > 0          -- high-damage edges are high-degree;
  (2) reldiff = |d_fix - d_rec|/d_fix decays with degree (O(1/d));
  (3) => the high-degree (high-vulnerability) edges have small reldiff: fixed ~ recompute.

Cora, full graph, 2 seeds. Outputs: results/exp_x4_highdeg.csv
Usage: .venv/bin/python scripts/exp_x4_highdeg_bulletproof.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.examples.ignn_cora import _load_cora
from exp_phase_transition import set_seed, train_ignn_kappa  # noqa: E402

SEEDS = [42, 137]
KAPPA = 0.90
BINS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)]
PER_BIN = 40


def normalize_from_raw(raw):
    N = raw.shape[0]
    AI = raw + torch.eye(N, device=raw.device, dtype=raw.dtype)
    c = AI.sum(1).clamp(min=1e-12).rsqrt()
    return c[:, None] * AI * c[None, :]


def reconverge(model, A, X_proj, z0, iters=600, tol=1e-9):
    ctx = {"A_hat": A.to(X_proj.dtype), "X_proj": X_proj}
    Z = z0.clone()
    with torch.no_grad():
        for _ in range(iters):
            Zn = model.operator(Z, ctx)
            if (Zn - Z).norm() < tol:
                break
            Z = Zn
    return Zn


def run_seed(data, device, seed):
    set_seed(seed)
    model, Z_star, ctx, _ = train_ignn_kappa(data, device, seed, KAPPA)
    A_hat = data["A_hat"].to(device)
    raw = (A_hat.abs() > 1e-10).float()
    raw.fill_diagonal_(0.0)
    deg = raw.sum(1)
    A_sc = normalize_from_raw(raw)                                  # self-consistent base
    X_proj = ctx["X_proj"]
    Z0 = reconverge(model, A_sc, X_proj, Z_star.clone())            # base equilibrium

    iu, ju = torch.where(torch.triu(raw > 0.5, diagonal=1))
    edges = list(zip(iu.tolist(), ju.tolist()))
    md_all = np.array([min(float(deg[i]), float(deg[j])) for (i, j) in edges])
    rng = np.random.default_rng(seed)
    sampled = []
    for (lo, hi) in BINS:
        ix = np.where((md_all >= lo) & (md_all <= hi))[0]
        if len(ix):
            sampled.extend(int(k) for k in rng.choice(ix, size=min(PER_BIN, len(ix)), replace=False))

    rows = []
    for k in sampled:
        i, j = edges[k]
        Af = A_sc.clone(); Af[i, j] = 0.0; Af[j, i] = 0.0
        dfix = float((reconverge(model, Af, X_proj, Z0) - Z0).norm())
        raw2 = raw.clone(); raw2[i, j] = 0.0; raw2[j, i] = 0.0
        drec = float((reconverge(model, normalize_from_raw(raw2), X_proj, Z0) - Z0).norm())
        rows.append({"seed": seed, "i": i, "j": j,
                     "min_deg": min(float(deg[i]), float(deg[j])),
                     "d_fix": dfix, "d_rec": drec})
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    data = _load_cora(Path("datasets/cora"))
    print(f"Cora full graph: N={data['N']}", flush=True)
    rows, t0 = [], time.time()
    for seed in SEEDS:
        print(f"[seed {seed}] ...", flush=True)
        rows.extend(run_seed(data, device, seed))
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out = Path("results/exp_x4_highdeg.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["seed", "i", "j", "min_deg", "d_fix", "d_rec"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {out}", flush=True)

    md = np.array([r["min_deg"] for r in rows])
    dfix = np.array([r["d_fix"] for r in rows])
    drec = np.array([r["d_rec"] for r in rows])
    ok = dfix > 1e-8
    md, dfix, drec = md[ok], dfix[ok], drec[ok]
    reldiff = np.abs(dfix - drec) / dfix

    print("\n" + "=" * 64)
    print(f"n edges = {len(md)}  (degree range {int(md.min())}-{int(md.max())})")
    # (1) damage correlates with degree
    print(f"(1) corr(damage d_fix, degree): Spearman rho = {spearmanr(dfix, md).correlation:.3f} "
          f"(expect > 0: high-damage edges are high-degree)")
    # (2) reldiff decays with degree
    print(f"(2) corr(reldiff, degree): Spearman rho = {spearmanr(reldiff, md).correlation:.3f} "
          f"(expect < 0: O(1/d))")
    print("    reldiff by degree bin:")
    for (lo, hi) in BINS:
        m = (md >= lo) & (md <= hi)
        if m.sum():
            tag = f"{lo}-{hi}" if hi < 1e8 else f"{lo}+"
            print(f"      deg {tag:>5}:  n={int(m.sum())}  mean reldiff={reldiff[m].mean():.3f}")
    # (3) the high-vulnerability (top-decile damage) edges: degree + agreement
    top = dfix >= np.quantile(dfix, 0.90)
    print(f"(3) top-decile-damage edges: mean degree={md[top].mean():.1f} "
          f"(vs overall {md.mean():.1f}); mean reldiff={reldiff[top].mean():.3f} "
          f"(vs overall {reldiff.mean():.3f})")
    tau = kendalltau(dfix, drec).correlation
    print(f"    overall Kendall tau(d_fix, d_rec) on full-graph edges = {tau:.3f}")
    dmg_deg = spearmanr(dfix, md).correlation
    if dmg_deg > 0.2:
        print("\nConclusion: high-damage edges are high-degree (recompute correction negligible "
              "there) => fixed ~ recompute on the high-vulnerability edges.")
    else:
        print(f"\nConclusion: high-damage edges are LOW-degree (corr={dmg_deg:.2f}); the recompute "
              "correction is largest exactly on the damaging edges, so fixed-norm masking is NOT "
              "interchangeable with topology deletion on the edges that matter. The valid defense "
              "is the THREAT MODEL (AEGIS = continuous edge-weight; fixed-norm is exact). "
              "See docs/x4_deletion_normalization_findings.md sec.5.")


if __name__ == "__main__":
    sys.exit(main() or 0)
