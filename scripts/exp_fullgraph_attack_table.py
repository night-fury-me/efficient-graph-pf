"""Exp: FULL-GRAPH four-quadrant attack table (paper Table `tab:attack_full`).

Full-graph counterpart of `scripts/exp_full_attack_table.py`. Instead of the
50-node BFS ego-subgraph, every attack runs on the ENTIRE graph via the
matrix-free pipeline (`iem.scalable.ScalableSensitivity`: Neumann resolvent +
forward/backward AD JVP/VJP, no dense O((Nd)^2) S_c is ever formed).

Four attacks at budget eps=0.10 on an IGNN, datasets Cora/Citeseer/WikiCS:
  1. SVD (AEGIS)  : dA* = eps * sym(v_1), v_1 = leading right singular vector of
                    S_c obtained matrix-free via ScalableSensitivity.top_k_svd.
  2. Cls-PGD      : 50-step PGD on the classification (cross-entropy) loss.
  3. Shift-PGD    : 50-step PGD on the equilibrium-shift objective (IFT gradient
                    through the resolvent); "solver validation, not an
                    independent baseline".
  4. Random       : eps-scaled symmetric noise.

Damage metric (BYTE-IDENTICAL to exp_full_attack_table.py): reconverged
equilibrium displacement ||z*(A+dA) - z*(A)||, NOT sigma_1. The exact attack +
damage helpers (`reconverge`, `apply_perturbation`, `measure_attack`, `set_seed`,
the subgraph `train_ignn`, SEEDS, EPS) are IMPORTED from exp_full_attack_table.py
so the four-attack semantics and the damage definition cannot drift. The ONLY
differences vs the subgraph script:
  (a) no ego-subgraph extraction -- full A_hat / Z_star / ctx from model(X, A);
  (b) the SVD direction comes from top_k_svd(Vh[0]) instead of dense
      torch.linalg.svd(S_c)[Vh][0] (same edge basis: triu i<j active-edge order,
      identical to constrained_sensitivity_matrix);
  (c) Cls-PGD / Shift-PGD build the perturbed adjacency with a VECTORIZED edge
      scatter that is bit-for-bit identical to the subgraph script's per-edge
      loop (verified: max|loop - vectorized| = 0.0), required because the loop's
      per-edge full-matrix .clone() is O(|E| N^2)/step and intractable at
      N~2700. The math (gradient, projection, budget) is unchanged.

Bug routed around (known codebase issue): ScalableSensitivity._estimate_rho runs
only 30 power-iteration steps and reports the operator 2-norm ||J_z v||, which
upper-bounds and can OVERSHOOT the spectral radius; when it returns >=1 the
Neumann depth silently pins to K=500 (under-truncating at high rho and producing
an inaccurate v_1). We therefore compute each dataset's rho independently with a
200-step power iteration + Rayleigh-quotient estimate (sign-aware, converges to
the true dominant eigenvalue), REPORT it, and if rho>=0.98 rebuild the operator
with neumann_terms=3000 so the SVD direction is accurate. (Cora full ~0.946.)
The damage metric itself is exact (reconverged) and unaffected by rho.

Seeds: SVD / Cls-PGD / Random use all 10 SEEDS. Shift-PGD is the expensive case
(50 PGD steps, each a VJP through the resolvent + a reconverge); if the full
10-seed x 3-dataset budget for Shift-PGD is prohibitive it is reduced to the
first N_SHIFT_SEEDS seeds (default 5) and this is stated in the output and CSV.
Pass --shift-seeds N to override (10 = full).

Output: results/fullgraph_attack_table.csv  (+ console mean+/-sd table)

Usage:
    .venv/bin/python scripts/exp_fullgraph_attack_table.py [--shift-seeds N] \
        [--datasets Cora,Citeseer,WikiCS]
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the EXACT attack/damage/train helpers from the subgraph script so the
# four-attack + damage semantics are identical; we only change the graph scope.
from scripts.exp_full_attack_table import (
    EPS_VALUES,
    SEEDS,
    apply_perturbation,
    load_datasets,
    measure_attack,
    reconverge,
    set_seed,
    train_ignn,
)
from iem.scalable import ScalableSensitivity, extract_ignn_weight

EPS = 0.10  # paper table budget for tab:attack_full
assert EPS in EPS_VALUES, "EPS must be one of the subgraph script's budgets"

DATASETS = ["Cora", "Citeseer", "WikiCS"]
N_SHIFT_SEEDS_DEFAULT = 5  # reduced Shift-PGD seed count if full budget too costly

# Randomized-SVD settings for the matrix-free top_k_svd (matches the full-graph
# defense experiment's range-finder settings).
SVD_K = 6
SVD_POWER_ITER = 7
SVD_SEED = 0  # fixed sketch seed for a deterministic v_1 per (model, graph)


# ----------------------------------------------------------------------------
# Vectorized perturbed-adjacency builder. Bit-identical to the subgraph script's
# per-edge loop (max|loop - this| = 0.0, verified), but O(|E|) not O(|E| N^2).
# Differentiable in `delta`, so it can replace the in-PGD-loop construction.
# ----------------------------------------------------------------------------
def apply_perturbation_vec(A, rows, cols, weights):
    """Symmetric edge-weight perturbation via scatter. Identical to
    apply_perturbation(A, edge_list, weights) but vectorized + differentiable.
    rows, cols are LongTensors of the i<j endpoints (same order as edge_list)."""
    A_pert = A.clone()
    A_pert[rows, cols] = A[rows, cols] + weights
    A_pert[cols, rows] = A[cols, rows] + weights
    return A_pert


def rho_rayleigh(op: ScalableSensitivity, iters: int = 200) -> float:
    """Spectral radius of J_z via power iteration + Rayleigh quotient.

    J_z is the (fixed, linearised-at-z*) operator Jacobian. Power iteration
    converges to the dominant eigenvector v; the Rayleigh quotient <v, J_z v>
    is the dominant eigenvalue (sign-aware), so |.| is the spectral radius.
    Robust where ScalableSensitivity._estimate_rho (30 iters, returns ||J_z v||)
    overshoots / under-converges.
    """
    torch.manual_seed(0)
    v = torch.randn(op.D, device=op.device, dtype=op.dtype)
    v = v / v.norm()
    for _ in range(iters):
        Jv = op._jvp_Jz(v)
        nv = Jv.norm()
        if nv < 1e-12:
            return 0.0
        v = Jv / nv
    return abs(float((v * op._jvp_Jz(v)).sum().item()))


def build_op(model, X, A, rho_thresh: float = 0.98):
    """Forward to the full-graph fixed point, build a matrix-free
    ScalableSensitivity, compute a trustworthy rho, and (if rho>=thresh) rebuild
    with deep Neumann truncation so v_1 is accurate. Returns (op, rho, rebuilt).

    For an IGNN operator the four Jacobian applications are routed through the
    CLOSED-FORM path (``ignn_weight=W``), verified ==autograd to machine precision
    by ``scripts/_verify_opt_b_analytic.py``. This removes the N x N backward graph
    that OOM'd full-graph Pubmed (>24 GB) and is ~19x faster at N=2708; it falls
    back to autograd for any non-IGNN operator (guarded by duck-typing)."""
    def F_op(z, c):
        return model.operator(z, c)

    with torch.no_grad():
        _, Z_star, ctx = model(X, A)
    ignn_W = (extract_ignn_weight(model)
              if hasattr(model, "_W_eff") and hasattr(model, "W") else None)
    op = ScalableSensitivity(F_op, Z_star, ctx, ignn_weight=ignn_W)
    rho = rho_rayleigh(op)
    rebuilt = False
    if rho >= rho_thresh:
        # Route around the K=500 pin: deep Neumann so the resolvent (and hence
        # v_1) is accurate at high spectral radius.
        op = ScalableSensitivity(F_op, Z_star, ctx, neumann_terms=3000,
                                 ignn_weight=ignn_W)
        rebuilt = True
    return op, Z_star, ctx, rho, rebuilt


def svd_direction(op: ScalableSensitivity):
    """Leading right singular vector v_1 of S_c (edge-space unit vector) and
    sigma_1, matrix-free. Same edge basis as constrained_sensitivity_matrix."""
    torch.manual_seed(SVD_SEED)
    _, sigma, Vh = op.top_k_svd(k=min(SVD_K, op.num_edges),
                                n_power_iter=SVD_POWER_ITER)
    return Vh[0].detach(), float(sigma[0])


def pgd_attack_fullgraph(model, Z_clean, ctx, rows, cols, n_edges, epsilon,
                         objective, y=None, n_steps=50):
    """PGD identical to exp_full_attack_table.pgd_attack EXCEPT the perturbed
    adjacency is built with the vectorized (bit-identical) scatter instead of a
    per-edge full-matrix-clone loop. Same init, step size, sign-step, per-coord
    clamp, L2 ball projection, inner reconverge loop, and objective sign."""
    A = ctx["A_hat"]
    step_size = epsilon / 10.0
    # Same init as the subgraph script (small random delta inside the ball).
    delta_init = torch.randn(n_edges, device=A.device) * (epsilon * 0.01)
    norm_init = delta_init.norm()
    if norm_init > epsilon:
        delta_init = delta_init * (epsilon / norm_init)
    delta = delta_init.requires_grad_(True)

    for _step in range(n_steps):
        A_pert = apply_perturbation_vec(A, rows, cols, delta)  # differentiable
        ctx_pert = {**ctx, "A_hat": A_pert}

        Z = Z_clean.detach().clone()
        with torch.enable_grad():
            for _ in range(50):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).detach().norm() < 1e-7:
                    break
                Z = Z_new

            if objective == "classification":
                logits = model.head(Z_new)
                loss = -F_func.cross_entropy(logits, y)
            else:  # equilibrium shift (IFT gradient through the unrolled solve)
                diff = Z_new - Z_clean.detach()
                loss = -diff.pow(2).sum()

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.data -= step_size * grad.sign()
            delta.data.clamp_(-epsilon / (n_edges ** 0.5), epsilon / (n_edges ** 0.5))
            norm = delta.data.norm()
            if norm > epsilon:
                delta.data *= epsilon / norm
        delta = delta.detach().requires_grad_(True)

    return delta.detach()


def run_single(ds_name, data, seed, eps, device, do_shift: bool):
    """One (dataset, seed) full-graph row. SVD/Cls/Random always; Shift only if
    do_shift. Returns dict (shift fields None when skipped)."""
    model = train_ignn(data, device, seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    op, Z_star, ctx, rho, rebuilt = build_op(model, X, A_hat)
    n_edges = op.num_edges
    if n_edges == 0:
        return None
    rows = op._edge_idx[:, 0].contiguous()
    cols = op._edge_idx[:, 1].contiguous()
    edge_list = op.edge_list  # for the byte-identical apply_perturbation calls

    with torch.no_grad():
        preds_clean = model.head(Z_star).argmax(dim=1)
    n_nodes = X.shape[0]

    # --- 1. AEGIS SVD (one-query, matrix-free) ---
    v1, sigma1 = svd_direction(op)
    svd_weights = eps * v1  # edge-space ||.||_2 == eps, same as subgraph script
    A_svd = apply_perturbation(A_hat, edge_list, svd_weights)
    dmg_svd, flips_svd = measure_attack(model, Z_star, ctx, A_svd, preds_clean)

    # --- 2. Classification-loss PGD (50 steps) ---
    delta_cls = pgd_attack_fullgraph(model, Z_star, ctx, rows, cols, n_edges, eps,
                                     objective="classification", y=y)
    A_cls = apply_perturbation(A_hat, edge_list, delta_cls)
    dmg_cls, flips_cls = measure_attack(model, Z_star, ctx, A_cls, preds_clean)

    # --- 4. Random perturbation (compute before optional Shift) ---
    rand_weights = torch.randn(n_edges, device=A_hat.device)
    rand_weights = rand_weights / rand_weights.norm() * eps
    A_rand = apply_perturbation(A_hat, edge_list, rand_weights)
    dmg_rand, flips_rand = measure_attack(model, Z_star, ctx, A_rand, preds_clean)

    # --- 3. Equilibrium-shift PGD (optional; expensive) ---
    if do_shift:
        delta_shift = pgd_attack_fullgraph(model, Z_star, ctx, rows, cols, n_edges,
                                           eps, objective="shift")
        A_shift = apply_perturbation(A_hat, edge_list, delta_shift)
        dmg_shift, flips_shift = measure_attack(model, Z_star, ctx, A_shift, preds_clean)
    else:
        dmg_shift, flips_shift = None, None

    row = {
        "dataset": ds_name, "seed": seed, "epsilon": eps,
        "n_nodes": n_nodes, "n_edges": n_edges,
        "rho": rho, "neumann_rebuilt": int(rebuilt), "neumann_K": op.neumann_K,
        "sigma1": sigma1,
        "dmg_svd": dmg_svd, "dmg_cls_pgd": dmg_cls,
        "dmg_shift_pgd": dmg_shift, "dmg_random": dmg_rand,
        "flips_svd": flips_svd, "flips_cls_pgd": flips_cls,
        "flips_shift_pgd": flips_shift, "flips_random": flips_rand,
        "fliprate_svd": flips_svd / n_nodes,
        "fliprate_cls_pgd": flips_cls / n_nodes,
        "fliprate_shift_pgd": (flips_shift / n_nodes) if do_shift else None,
        "fliprate_random": flips_rand / n_nodes,
        "cls_pgd_over_svd_dmg": dmg_cls / max(dmg_svd, 1e-10),
        "shift_pgd_over_svd_dmg": (dmg_shift / max(dmg_svd, 1e-10)) if do_shift else None,
    }
    del op
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-seeds", type=int, default=N_SHIFT_SEEDS_DEFAULT,
                    help="number of seeds (from the front of SEEDS) for Shift-PGD")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    n_shift = max(0, min(args.shift_seeds, len(SEEDS)))
    shift_seeds = set(SEEDS[:n_shift])
    datasets_run = [d.strip() for d in args.datasets.split(",") if d.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    print(f"Device: {device} | eps={EPS} | full graph (no ego-subgraph)", flush=True)
    print(f"SVD/Cls-PGD/Random: {len(SEEDS)} seeds | "
          f"Shift-PGD: {n_shift} seeds {sorted(shift_seeds)}", flush=True)

    datasets = load_datasets()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rows = []
    for ds_name in datasets_run:
        data = datasets[ds_name]
        for seed_idx, seed in enumerate(SEEDS):
            do_shift = seed in shift_seeds
            t0 = time.time()
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)}) "
                  f"shift={'Y' if do_shift else 'n'}", end=" ", flush=True)
            try:
                r = run_single(ds_name, data, seed, EPS, device, do_shift)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    gc.collect(); torch.cuda.empty_cache()
                    print("OOM-SKIP", flush=True)
                    continue
                raise
            if r is None:
                print("SKIP(no edges)", flush=True)
                continue
            rows.append(r)
            shift_str = (f"ShiftPGD={r['dmg_shift_pgd']:.3f}"
                         if r["dmg_shift_pgd"] is not None else "ShiftPGD=--")
            print(f"rho={r['rho']:.3f} K={r['neumann_K']} sig1={r['sigma1']:.1f} | "
                  f"SVD={r['dmg_svd']:.3f}({r['flips_svd']}f) "
                  f"ClsPGD={r['dmg_cls_pgd']:.3f}({r['flips_cls_pgd']}f) "
                  f"{shift_str} Rand={r['dmg_random']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    if not rows:
        print("No rows produced.")
        return 1

    # --- CSV ---
    csv_path = results_dir / "fullgraph_attack_table.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n", flush=True)

    # --- Summary: equilibrium damage (mean +/- sd) ---
    print("=" * 132)
    print(f"FULL-GRAPH ATTACK TABLE (tab:attack_full): EQUILIBRIUM DAMAGE "
          f"||z*(A+dA)-z*(A)||  (mean +/- sd), eps={EPS}")
    print("=" * 132)
    hdr = (f"{'Dataset':<10} {'N':>6} {'|E|':>7} {'rho':>7} "
           f"{'AEGIS SVD':>18} {'Cls-PGD':>18} {'Shift-PGD':>18} {'Random':>16} "
           f"{'SVD>=Cls?':>10}")
    print(hdr)
    print("-" * 132)
    for ds_name in datasets_run:
        sub = [r for r in rows if r["dataset"] == ds_name]
        if not sub:
            continue
        N = sub[0]["n_nodes"]; E = sub[0]["n_edges"]
        rho_m = np.mean([r["rho"] for r in sub])

        def cell(key):
            m, s, n = _agg([r[key] for r in sub])
            return f"{m:.3f}+/-{s:.3f}(n{n})"

        m_svd = _agg([r["dmg_svd"] for r in sub])[0]
        m_cls = _agg([r["dmg_cls_pgd"] for r in sub])[0]
        m_shift = _agg([r["dmg_shift_pgd"] for r in sub])[0]
        # headline: does one-query SVD match/exceed 50-step PGD baselines?
        beats = "YES" if (m_svd >= m_cls and (np.isnan(m_shift) or m_svd >= m_shift)) else "NO"
        print(f"{ds_name:<10} {N:>6} {E:>7} {rho_m:>7.3f} "
              f"{cell('dmg_svd'):>18} {cell('dmg_cls_pgd'):>18} "
              f"{cell('dmg_shift_pgd'):>18} {cell('dmg_random'):>16} {beats:>10}")

    # --- Summary: prediction flip rate % ---
    print()
    print("=" * 132)
    print("FULL-GRAPH ATTACK TABLE: PREDICTION FLIP RATE %  (mean +/- sd)")
    print("=" * 132)
    print(f"{'Dataset':<10} {'AEGIS SVD':>18} {'Cls-PGD':>18} "
          f"{'Shift-PGD':>18} {'Random':>16}")
    print("-" * 132)
    for ds_name in datasets_run:
        sub = [r for r in rows if r["dataset"] == ds_name]
        if not sub:
            continue

        def pct(key):
            m, s, n = _agg([(r[key] * 100 if r[key] is not None else None) for r in sub])
            return f"{m:.1f}+/-{s:.1f}(n{n})"

        print(f"{ds_name:<10} {pct('fliprate_svd'):>18} {pct('fliprate_cls_pgd'):>18} "
              f"{pct('fliprate_shift_pgd'):>18} {pct('fliprate_random'):>16}")

    print(f"\nShift-PGD seed count: {n_shift}/{len(SEEDS)} "
          f"({'FULL' if n_shift == len(SEEDS) else 'REDUCED for cost'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
