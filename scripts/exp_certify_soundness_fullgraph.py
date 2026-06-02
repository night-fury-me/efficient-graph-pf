#!/usr/bin/env python
# ---------------------------------------------------------------------------
# FULL-GRAPH soundness spot-check for the AEGIS-Certify certified radius.
#
# Both prior soundness gates (pilot + tighten) ran the worst-case attack only on
# DENSE ego-subgraphs, with the full-graph claim resting on the editorial
# argument that the sqrt(N)-drop is a strict reduction of an over-bound and the
# rmatvec == dense-projection equivalence. This script CLOSES that gap: it
# rebuilds the worst-case attack in the FULL edge space, applies it to the FULL
# A_hat, RECONVERGES the full-graph equilibrium, and checks whether the target
# node's argmax flips.
#
# SHIPPING constant: T3 = 2-hop-local curvature
#     L_{J,v} = ||W||_2^2 * max_{u in 2hop(v)} ||z*_u||_2
# (exactly as in scripts/exp_certify_tighten.py).
#
# Certified radius (per node v):
#     rho_v = min_c  positive-root( m_v^(c) - L1_c r - C_v r^2 = 0 )
#     L1_c  = ||(W_{y_v}-W_c) S_{c,v}||_2 / sqrt(2)           (unit-basis edge norm)
#     C_v   = ||W_{y_v}-W_c||_2 * (1-kappa)^{-2} * L_{J,v} / 2
#
# SOUNDNESS CLAIM: no symmetric edge-supported delta-Ahat with
#     ||delta-Ahat||_F < rho_v  may flip node v  -- now verified with full-graph
# reconvergence. A BREACH is a flip at ||delta||_F = 0.99 * rho_v.
#
# The machinery (rho_v rmatvec, S_{c,v} VJP, worst-case direction, symmetric dA
# builder, divergence-capped reconverge) is REUSED verbatim from
# exp_certify_tighten.py and exp_fullgraph_attack_table.py.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.revision_R2._common import load_dataset, train_ignn  # noqa: E402
from iem.scalable import ScalableSensitivity  # noqa: E402
from scripts.exp_fullgraph_attack_table import build_op, rho_rayleigh  # noqa: E402
from scripts.exp_certify_tighten import (  # noqa: E402
    F_op_factory,
    w_norm_2,
    positive_root,
    node_norms,
    _build_symmetric_dA,
    _safe_reconverge,
    _two_hop_for_nodes,
    SQRT2,
)

RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Full-graph rho_v (T3) for one node, matrix-free. Same math as
# certify_fullgraph(): L1_c = ||rmatvec(u)|| / sqrt(2), C_v from L_{J,v}.
# Returns (rho_v, binding_c, margin, edge_resp_binding) so the soundness attack
# can reuse the SAME rmatvec response (the worst-case edge direction) without a
# second VJP.
# ---------------------------------------------------------------------------
def fullgraph_rho_for_node(model, op, Z_star, logits, preds, v, kappa, LJ_v):
    dev = Z_star.device
    d = model.hidden
    D = op.D
    W = model.head.weight.detach()
    kfac = (1.0 - kappa) ** (-2)

    p = int(preds[v])
    marg = (logits[v, p] - logits[v])
    best_rho = float("inf")
    best_c = -1
    best_m = 0.0
    best_resp = None
    for c in range(logits.shape[1]):
        if c == p:
            continue
        m_c = float(marg[c])
        if m_c <= 0:
            continue
        wgap = (W[p] - W[c])
        u = torch.zeros(D, device=dev, dtype=Z_star.dtype)
        u[v * d:(v + 1) * d] = wgap
        edge_resp = op.rmatvec(u)                     # S_{c,v}^T (W_p - W_c)  in R^|E|
        L1_c = float(edge_resp.norm()) / SQRT2
        wg_norm = float(wgap.norm())
        C_v = wg_norm * kfac * LJ_v / 2.0
        r_c = positive_root(L1_c, C_v, m_c)
        if r_c < best_rho:
            best_rho = r_c
            best_c = c
            best_m = m_c
            best_resp = edge_resp
    return best_rho, best_c, best_m, best_resp


# ---------------------------------------------------------------------------
# Full-graph worst-case attack at an exact Frobenius budget, with reconvergence.
#
# w_dir is the UNIT edge-space direction aligned with -S_{c,v}^T (W_p - W_c*)
# (the binding-competitor rmatvec response), i.e. the first-order steepest-
# descent direction on the margin m_v^(c*). _build_symmetric_dA maps it to a
# symmetric edge-supported dA with EXACT ||dA||_F = target_fro (the /sqrt(2)
# unit-basis correction lives inside _build_symmetric_dA). We then add dA to the
# FULL A_hat, reconverge the FULL equilibrium from Z_star, and check the argmax.
# ---------------------------------------------------------------------------
def fullgraph_attack_and_check(model, A_full, X_proj, Z_star, op, w_dir,
                               target_fro, v, p, max_iter=400, tol=1e-9):
    if target_fro <= 0:
        with torch.no_grad():
            lg = model.head(Z_star)[v]
        return int(lg.argmax()) != p, 0.0, 0.0, False, int(lg.argmax())
    dA = _build_symmetric_dA(op, w_dir, target_fro)
    true_fro = float(dA.norm())                       # MEASURED ||delta-Ahat||_F
    A_pert = (A_full + dA).contiguous()
    ctx_p = {"A_hat": A_pert, "X_proj": X_proj}
    Z_p, diverged, res = _safe_reconverge(model, Z_star, ctx_p,
                                          max_iter=max_iter, tol=tol)
    if diverged:
        # A diverged perturbed system destroys the clean class -> counts as flip.
        return True, true_fro, float("inf"), True, -1
    with torch.no_grad():
        lg = model.head(Z_p)[v]
    new_pred = int(lg.argmax())
    return new_pred != p, true_fro, res, False, new_pred


def stratified_sample(rho_by_node, n_target, rho_min):
    """Pick ~n_target nodes spanning the rho_v range, stratified by quantile,
    requiring rho_v > rho_min so each test is non-trivial. rho_by_node is a list
    of (node, rho, binding_c, margin, edge_resp)."""
    elig = [t for t in rho_by_node if t[1] > rho_min and t[2] >= 0
            and t[4] is not None and float(t[4].norm()) > 1e-30]
    if not elig:
        return []
    elig.sort(key=lambda t: t[1])                     # ascending rho
    M = len(elig)
    if M <= n_target:
        return elig
    # even quantile spacing across the sorted-by-rho eligible set
    idxs = [round(k * (M - 1) / (n_target - 1)) for k in range(n_target)]
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(elig[i])
    return out


def run_dataset_seed(ds, seed, dev, args, writer, log):
    X, A, y, tm, nf, nc = load_dataset(ds)
    X, A, y = X.to(dev), A.to(dev), y.to(dev)
    model = train_ignn(X, A, y, tm, nf, nc, dev, seed=seed, epochs=args.epochs)

    # Full-graph op (deep-Neumann rebuild if kappa>=0.98), equilibrium, ctx.
    op, Z_star, ctx, kappa, rebuilt = build_op(model, X, A)
    A_full = ctx["A_hat"].detach()
    X_proj = ctx["X_proj"].detach()
    N = X.shape[0]

    with torch.no_grad():
        logits = model.head(Z_star)
    preds = logits.argmax(dim=1)
    acc = float((preds.cpu() == y.cpu()).float().mean())

    # clean-equilibrium residual sanity
    with torch.no_grad():
        eqres = float((model.operator(Z_star, ctx) - Z_star).norm())

    # T3 (shipping constant) per-node curvature for ALL correctly-classified nodes
    nrm = node_norms(Z_star)
    w2 = w_norm_2(model) ** 2
    corr_nodes = (preds.cpu() == y.cpu()).nonzero(as_tuple=True)[0].tolist()
    LJ_map = _two_hop_for_nodes(nrm, op.edge_list, N, corr_nodes, w2)

    print(f"[{ds}/full/s{seed}] N={N} E={op.num_edges} kappa={kappa:.4f} "
          f"rebuilt={rebuilt} acc={acc:.3f} eqres={eqres:.1e} "
          f"correct={len(corr_nodes)} ||z*||_F={float(Z_star.norm()):.2f} "
          f"max||z*_i||={float(nrm.max()):.2f}", flush=True)

    # Step 1: rho_v (T3) for every correctly-classified node
    rho_by_node = []
    for v in corr_nodes:
        rho, bc, m, resp = fullgraph_rho_for_node(
            model, op, Z_star, logits, preds, v, kappa, LJ_map[int(v)])
        if bc >= 0:
            rho_by_node.append((int(v), rho, bc, m, resp))

    pos = [t[1] for t in rho_by_node if t[1] > 0]
    pos.sort()
    med = pos[len(pos) // 2] if pos else float("nan")
    print(f"    rho_v(T3): n_pos={len(pos)} median={med:.4f} "
          f"max={(pos[-1] if pos else 0):.4f} "
          f"n>{args.rho_min}={sum(1 for r in pos if r > args.rho_min)}", flush=True)

    # Step 2: stratified sample of ~args.sample certified nodes (rho>rho_min)
    chosen = stratified_sample(rho_by_node, args.sample, args.rho_min)
    if not chosen:
        print(f"    NO eligible nodes (rho>{args.rho_min}); skipping.", flush=True)
        return
    print(f"    sampled {len(chosen)} nodes, rho range "
          f"[{chosen[0][1]:.4f}, {chosen[-1][1]:.4f}]", flush=True)

    # Steps 3-4: worst-case full-graph attack at 0.99*rho and 1.5*rho
    n_breach_below = 0
    n_flip_above = 0
    max_res_below = 0.0
    worst_fro_err = 0.0
    diverged_below = 0
    for (v, rho, bc, m, resp) in chosen:
        p = int(preds[v])
        gn = float(resp.norm())
        w_dir = -(resp / gn)                          # steepest-descent on margin

        tgt_below = args.frac_below * rho
        flip_b, fro_b, res_b, div_b, np_b = fullgraph_attack_and_check(
            model, A_full, X_proj, Z_star, op, w_dir, tgt_below, v, p,
            max_iter=args.max_iter, tol=args.tol)
        if math.isfinite(res_b):
            max_res_below = max(max_res_below, res_b)
        worst_fro_err = max(worst_fro_err, abs(fro_b - tgt_below) / (tgt_below + 1e-30))
        if div_b:
            diverged_below += 1
        if flip_b:
            n_breach_below += 1

        tgt_above = args.frac_above * rho
        flip_a, fro_a, res_a, div_a, np_a = fullgraph_attack_and_check(
            model, A_full, X_proj, Z_star, op, w_dir, tgt_above, v, p,
            max_iter=args.max_iter, tol=args.tol)
        if flip_a:
            n_flip_above += 1

        writer.writerow({
            "dataset": ds, "seed": seed, "node": v, "pred": p, "binding_c": bc,
            "rho_v": rho, "margin": m,
            "fro_below": fro_b, "true_fro_err_below": abs(fro_b - tgt_below),
            "breach_below_rho": int(flip_b), "diverged_below": int(div_b),
            "new_pred_below": np_b, "reconv_res_below": res_b,
            "flip_above_rho": int(flip_a), "fro_above": fro_a,
            "reconv_res_above": res_a, "kappa": kappa, "n_edges": op.num_edges,
        })

    summary = {
        "dataset": ds, "seed": seed, "n_tested": len(chosen),
        "n_breach_below_rho": n_breach_below,
        "n_flip_above_rho": n_flip_above,
        "frac_flip_above": n_flip_above / len(chosen),
        "max_reconv_res_below": max_res_below,
        "max_true_fro_relerr": worst_fro_err,
        "diverged_below": diverged_below,
        "kappa": kappa,
    }
    log.append(summary)
    print(f"    >>> tested={len(chosen)} BREACH<rho={n_breach_below} "
          f"flip@{args.frac_above}rho={n_flip_above} "
          f"({summary['frac_flip_above']*100:.0f}%) "
          f"max_reconv_res={max_res_below:.1e} "
          f"max_fro_relerr={worst_fro_err:.1e} diverged<rho={diverged_below}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Cora,Citeseer")
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--sample", type=int, default=25,
                    help="approx certified nodes to attack per (dataset,seed)")
    ap.add_argument("--rho-min", type=float, default=0.02,
                    help="require rho_v > this so the test is non-trivial")
    ap.add_argument("--frac-below", type=float, default=0.99)
    ap.add_argument("--frac-above", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max-iter", type=int, default=400,
                    help="reconverge iterations (divergence-capped)")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    datasets = [s for s in args.datasets.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    out_csv = open(RESULTS / "certify_soundness_fullgraph.csv", "w", newline="")
    writer = csv.DictWriter(out_csv, fieldnames=[
        "dataset", "seed", "node", "pred", "binding_c", "rho_v", "margin",
        "fro_below", "true_fro_err_below", "breach_below_rho", "diverged_below",
        "new_pred_below", "reconv_res_below", "flip_above_rho", "fro_above",
        "reconv_res_above", "kappa", "n_edges"])
    writer.writeheader()

    log = []
    for ds in datasets:
        for seed in seeds:
            run_dataset_seed(ds, seed, dev, args, writer, log)
            out_csv.flush()
    out_csv.close()

    total_tested = sum(s["n_tested"] for s in log)
    total_breach = sum(s["n_breach_below_rho"] for s in log)
    total_above = sum(s["n_flip_above_rho"] for s in log)

    print("\n================ FULL-GRAPH SOUNDNESS SUMMARY ================")
    for s in log:
        print(f"  {s['dataset']}/s{s['seed']}: tested={s['n_tested']} "
              f"BREACH<rho={s['n_breach_below_rho']} "
              f"flip@{args.frac_above}rho={s['n_flip_above_rho']} "
              f"({s['frac_flip_above']*100:.0f}%) "
              f"max_reconv_res={s['max_reconv_res_below']:.1e} "
              f"kappa={s['kappa']:.3f}")
    print(f"\n  TOTAL: breaches below rho_v = {total_breach} / {total_tested}")
    print(f"  TOTAL: flips at {args.frac_above}*rho = {total_above} / {total_tested} "
          f"({100*total_above/max(total_tested,1):.0f}%)")
    verdict = "SOUND (0 breaches under full-graph reconvergence)" if total_breach == 0 \
        else f"UNSOUND ({total_breach} breaches!)"
    print(f"  VERDICT: {verdict}")
    print(f"\nDONE in {time.time()-t0:.1f}s")
    print("Wrote results/certify_soundness_fullgraph.csv")


if __name__ == "__main__":
    main()
