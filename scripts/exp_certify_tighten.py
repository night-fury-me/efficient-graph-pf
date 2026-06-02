#!/usr/bin/env python3
"""Curvature-constant TIGHTENING for the AEGIS-Certify per-node radius rho_v.

Builds directly on the validated pilot (`scripts/exp_certify_pilot.py`,
findings `paper/review/certify_pilot_findings.md`). The pilot is SOUND but its
full-graph non-vacuity COLLAPSES because the second-order curvature constant

    L_J = ||W||_2^2 * ||z*||_F                                    (T2, loose control)

uses the WHOLE-GRAPH Frobenius norm ||z*||_F ~ O(sqrt(N)). For a PER-NODE radius
that is far too loose. The per-node curvature is governed by the OPERATOR norm of
J_A, set by the largest single-node embedding, so the candidate tightenings are

    L_J     = ||W||_2^2 * max_i ||z*_i||_2                        (T1, drops sqrt(N))
    L_{J,v} = ||W||_2^2 * max_{u in 2-hop(v)} ||z*_u||_2          (T3, per-node local)

This script recomputes rho_v under {T1, T2, T3}, measures non-vacuity, and
RE-RUNS THE SOUNDNESS GATE UNCHANGED (worst-case first-order edge attack at
0.99*rho_v + random symmetric directions, reconverge, count flips). A candidate
is ACCEPTED only if breaches == 0 on ALL datasets/seeds.

SOUNDNESS LOGIC (why a tighter L_J can stay sound)
--------------------------------------------------
The curvature term bounds the second-order remainder of the per-node margin under
the constrained (symmetric edge-supported) perturbation. The relevant Jacobian
J_A is applied to the equilibrium z* and its operator norm is bounded by
||W||_2^2 * max_i ||z*_i||_2 (the per-node embedding magnitude), NOT the Frobenius
sum. T2 is therefore a loose over-bound of T1 by a factor ||z*||_F / max_i||z*_i||
~ sqrt(N). T1 is still a valid UPPER bound on the curvature, so rho_v(T1) should
remain sound. The soundness gate is the arbiter: if T1 ever breaches, it is
rejected and we fall back to an intermediate (sqrt(deg) inflation) or T3.

EVERYTHING ELSE IS IDENTICAL TO THE PILOT:
  - L1_c = ||(W_{y_v}-W_c) S_{c,v}^{paper}||_2,  S^{paper}=S^{code}/sqrt(2)  (bug B1 fix)
  - C_v  = ||W_{y_v}-W_c||_2 * (1-kappa)^{-2} * L_J / 2
  - rho_v = positive root of  m_v^{(c)} - L1_c r - C_v r^2 = 0, min over c
  - kappa via rho_rayleigh (honest power iteration + Rayleigh quotient)
  - max-node-norm taken over the SAME equilibrium z* (Z_sub dense / Z_star full)
    that the rest of that node's computation uses.
  - the soundness attack measures the LITERAL Frobenius norm of delta-Ahat and
    reconverges with the same divergence cap.

Outputs:
    results/certify_tighten_dense.csv       per-node rho_v / r_v, per candidate (dense exact)
    results/certify_tighten_fullgraph.csv   per-node rho_v / r_v, per candidate (matrix-free)
    results/certify_tighten_soundness.csv   per-attacked-node breach records, per candidate
    results/certify_tighten_summary.csv     the deliverable table (non-vacuity + breaches)

Usage:
    .venv/bin/python scripts/exp_certify_tighten.py \
        [--seeds 42,137,271] [--datasets Cora,Citeseer,WikiCS] \
        [--dense-dataset Cora] [--dense-nodes 80] [--fullgraph-sample 300] \
        [--sound-nodes 24] [--candidates T1,T2,T3] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.revision_R2._common import load_dataset, train_ignn, reconverge  # noqa: E402
from iem.adversarial import (  # noqa: E402
    extract_ego_subgraph,
    structural_sensitivity_matrix,
    constrained_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity  # noqa: E402
from scripts.exp_fullgraph_attack_table import rho_rayleigh  # noqa: E402

SQRT2 = math.sqrt(2.0)
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers (identical math to the pilot)
# ---------------------------------------------------------------------------
def F_op_factory(model):
    def F_op(z, c):
        return model.operator(z, c)
    return F_op


def w_norm_2(model) -> float:
    """||W||_2 of the state-propagation weight (spectral-norm-parametrized)."""
    return float(torch.linalg.svdvals(model.W.weight.detach())[0])


def positive_root(L1: float, C: float, m: float) -> float:
    """Positive root r of  m - L1 r - C r^2 = 0  (m>0, C>=0, L1>=0)."""
    if m <= 0:
        return 0.0
    if C <= 1e-30:
        return m / (L1 + 1e-30)
    disc = L1 * L1 + 4.0 * C * m
    return (-L1 + math.sqrt(disc)) / (2.0 * C)


# ---------------------------------------------------------------------------
# Curvature-constant candidates
# ---------------------------------------------------------------------------
def node_norms(Z: torch.Tensor) -> torch.Tensor:
    """Per-node embedding 2-norms ||z*_i||_2 for Z of shape (N, d)."""
    return Z.norm(dim=1)


def two_hop_max_norm(Z: torch.Tensor, edge_pairs, N: int) -> torch.Tensor:
    """For each node v, max_{u in CLOSED 2-hop(v)} ||z*_u||_2.

    edge_pairs: iterable of (i, j) undirected existing edges (i<j fine).
    Builds the symmetric 1-hop adjacency (bool), squares the reachability to get
    the closed 2-hop mask (includes v and all u within graph distance <= 2),
    then maxes the per-node norm over each node's 2-hop set. Self is always
    included so the result is >= ||z*_v|| (the local constant never drops below
    the node's own magnitude). Done in chunks to stay memory-safe on full graphs.
    """
    dev = Z.device
    nn_ = node_norms(Z)                                   # (N,)
    if N == 0:
        return nn_
    ei = torch.as_tensor(list(edge_pairs), device=dev, dtype=torch.long)
    # adjacency list as a sparse boolean via index buckets
    # neighbors[i] -> set of 1-hop neighbors (both directions); include self.
    rows = [[] for _ in range(N)]
    if ei.numel() > 0:
        for a, b in ei.tolist():
            rows[a].append(b)
            rows[b].append(a)
    out = nn_.clone()
    for v in range(N):
        # closed 2-hop = self U N(v) U N(N(v))
        hop = {v}
        for u in rows[v]:
            hop.add(u)
            for w in rows[u]:
                hop.add(w)
        idx = torch.as_tensor(sorted(hop), device=dev, dtype=torch.long)
        out[v] = nn_[idx].max()
    return out


def make_LJ_provider(name: str, model, Z: torch.Tensor, edge_pairs, N: int):
    """Return (description, LJ_fn) where LJ_fn(v:int)->float gives L_{J} (or
    L_{J,v}) for node v under the named candidate, using the SAME equilibrium Z
    everywhere. T1/T2 are node-independent (constant fn); T3 is per-node.

    name in {T1, T2, T3} or {T1d<mult>} intermediate (T1 inflated by a constant
    or by sqrt(max-degree)).
    """
    w2 = w_norm_2(model) ** 2
    nn_ = node_norms(Z)
    if name == "T2":  # loose Frobenius control
        val = w2 * float(Z.norm())
        return ("L_J = ||W||^2 * ||z*||_F  (Frobenius, control)", lambda v: val)
    if name == "T1":  # operator-consistent: drop sqrt(N)
        val = w2 * float(nn_.max())
        return ("L_J = ||W||^2 * max_i ||z*_i||_2", lambda v: val)
    if name == "T3":  # per-node 2-hop local
        loc = two_hop_max_norm(Z, edge_pairs, N) * w2     # (N,)
        return ("L_{J,v} = ||W||^2 * max_{u in 2-hop(v)} ||z*_u||_2",
                lambda v: float(loc[v]))
    if name == "T1sqrtdeg":  # intermediate: T1 inflated by sqrt(max node degree)
        # degree of each node from edge_pairs
        deg = torch.zeros(N, device=Z.device)
        for a, b in edge_pairs:
            deg[a] += 1; deg[b] += 1
        maxdeg = float(deg.max().clamp(min=1.0))
        val = w2 * float(nn_.max()) * math.sqrt(maxdeg)
        return (f"L_J = ||W||^2 * max_i||z*_i|| * sqrt(maxdeg={maxdeg:.0f})",
                lambda v: val)
    raise ValueError(f"unknown candidate {name}")


# ---------------------------------------------------------------------------
# (1a) DENSE exact rho_v on an IGNN ego-subgraph, for ONE candidate
# ---------------------------------------------------------------------------
def certify_dense(model, A_sub, ctx_sub, Z_sub, y_sub, kappa, cand, tag,
                  LJ_fn, LJ_desc, Sc, edge_list, writer):
    """Exact per-node rho_v / r_v under candidate `cand`. Sc is the precomputed
    paper-convention sensitivity (shared across candidates: only L_J changes)."""
    dev = Z_sub.device
    N = A_sub.shape[0]
    d = model.hidden
    W = model.head.weight.detach()                       # (C, d)
    kfac = (1.0 - kappa) ** (-2)
    E = len(edge_list)

    with torch.no_grad():
        logits = model.head(Z_sub)
    preds = logits.argmax(dim=1)
    Ccls = logits.shape[1]

    rho_v = torch.zeros(N, device=dev)
    r_v = torch.zeros(N, device=dev)
    bind_c = torch.full((N,), -1, dtype=torch.long, device=dev)
    margins = torch.zeros(N, device=dev)
    LJ_used = torch.zeros(N, device=dev)

    for v in range(N):
        p = int(preds[v])
        Sv = Sc[v * d:(v + 1) * d, :]                    # (d, |E|) paper conv
        marg = (logits[v, p] - logits[v])
        L_J_v = LJ_fn(v)                                 # candidate constant for v
        LJ_used[v] = L_J_v
        best_rho = float("inf"); best_r = float("inf"); best_c = -1
        for c in range(Ccls):
            if c == p:
                continue
            m_c = float(marg[c])
            if m_c <= 0:
                continue
            wgap = (W[p] - W[c])
            wg_norm = float(wgap.norm())
            L1_c = float((wgap @ Sv).norm())
            C_v = wg_norm * kfac * L_J_v / 2.0
            r_c = positive_root(L1_c, C_v, m_c)
            r_lin = m_c / (L1_c + 1e-30)
            if r_c < best_rho:
                best_rho = r_c; best_c = c
            if r_lin < best_r:
                best_r = r_lin
        if best_c < 0:
            rho_v[v] = 0.0; r_v[v] = 0.0
        else:
            rho_v[v] = best_rho; r_v[v] = best_r
            bind_c[v] = best_c; margins[v] = float(marg[best_c])

    correct = (preds.cpu() == y_sub.cpu())
    for v in range(N):
        if not bool(correct[v]):
            continue
        writer.writerow({
            "candidate": cand, "tag": tag, "node": v, "pred": int(preds[v]),
            "binding_c": int(bind_c[v]), "margin": float(margins[v]),
            "rho_v": float(rho_v[v]), "r_v_linear": float(r_v[v]),
            "ratio_rho_over_r": float(rho_v[v] / (r_v[v] + 1e-30)),
            "kappa": kappa, "L_J_node": float(LJ_used[v]), "n_nodes": N, "n_edges": E,
        })
    return {
        "rho_v": rho_v, "r_v": r_v, "bind_c": bind_c, "margins": margins,
        "preds": preds, "correct": correct, "d": d, "LJ_desc": LJ_desc,
    }


# ---------------------------------------------------------------------------
# (1b) MATRIX-FREE rho_v on the full graph, for ONE candidate
# ---------------------------------------------------------------------------
def certify_fullgraph(model, X, y, kappa, op, Z_star, cand, tag,
                      LJ_fn, sample_nodes, writer, acc=None):
    """Per-node rho_v via matrix-free rmatvec under candidate `cand`. If `acc`
    (a {"rho":[], "ratio":[]} dict) is given, the per-node rho/ratio are appended
    so the caller does not have to re-run the expensive rmatvec a second time."""
    dev = Z_star.device
    d = model.hidden
    D = op.D
    W = model.head.weight.detach()
    kfac = (1.0 - kappa) ** (-2)

    with torch.no_grad():
        logits = model.head(Z_star)
    preds = logits.argmax(dim=1)
    Ccls = logits.shape[1]

    n_rows = 0
    for v in sample_nodes:
        v = int(v)
        p = int(preds[v])
        marg = (logits[v, p] - logits[v])
        L_J_v = LJ_fn(v)
        best_rho = float("inf"); best_r = float("inf"); best_c = -1; best_m = 0.0
        for c in range(Ccls):
            if c == p:
                continue
            m_c = float(marg[c])
            if m_c <= 0:
                continue
            wgap = (W[p] - W[c])
            u = torch.zeros(D, device=dev, dtype=Z_star.dtype)
            u[v * d:(v + 1) * d] = wgap
            edge_resp = op.rmatvec(u)
            L1_c = float(edge_resp.norm()) / SQRT2
            wg_norm = float(wgap.norm())
            C_v = wg_norm * kfac * L_J_v / 2.0
            r_c = positive_root(L1_c, C_v, m_c)
            r_lin = m_c / (L1_c + 1e-30)
            if r_c < best_rho:
                best_rho = r_c; best_c = c; best_m = m_c
            if r_lin < best_r:
                best_r = r_lin
        if best_c < 0:
            continue
        corr = int(preds[v]) == int(y[v])
        writer.writerow({
            "candidate": cand, "tag": tag, "node": v, "pred": p,
            "binding_c": best_c, "margin": best_m,
            "rho_v": best_rho, "r_v_linear": best_r,
            "ratio_rho_over_r": best_rho / (best_r + 1e-30),
            "correct": int(corr), "kappa": kappa, "L_J_node": float(L_J_v),
            "n_nodes": int(X.shape[0]), "n_edges": op.num_edges,
        })
        if acc is not None:
            acc["rho"].append(best_rho)
            acc["ratio"].append(best_rho / (best_r + 1e-30))
        n_rows += 1
    return n_rows


# ---------------------------------------------------------------------------
# (2) SOUNDNESS — verbatim attack machinery from the pilot
# ---------------------------------------------------------------------------
def _build_symmetric_dA(op, w_edges: torch.Tensor, target_fro: float) -> torch.Tensor:
    """Symmetric edge-supported dA with EXACT Frobenius norm target_fro.
    op._edges_to_delta_A(v) gives ||dA||_F = sqrt(2)||v||_2, so feed w/sqrt(2)
    then rescale to remove drift."""
    dA = op._edges_to_delta_A(w_edges / SQRT2)
    fro = dA.norm()
    if float(fro) < 1e-30:
        return dA
    return dA * (target_fro / fro)


def _safe_reconverge(model, Z_init, ctx_p, max_iter=300, tol=1e-9):
    """Reconverge with a divergence guard. Returns (Z, diverged, residual)."""
    Z = Z_init.clone()
    base = max(float(Z.norm()), 1.0)
    res = float("nan")
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_p)
            nn_ = float(Z_new.norm())
            if not math.isfinite(nn_) or nn_ > 1e6 * base:
                return Z_new, True, float("inf")
            res = float((Z_new - Z).norm())
            if res < tol * max(float(Z.norm()), 1.0):
                return Z_new, False, res
            Z = Z_new
    return Z, False, res


def _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, target_fro,
                      v, p, c, W):
    """Apply symmetric dA of exact ||.||_F=target_fro along w_dir, reconverge,
    return (flipped?, margin_after, true_fro, residual). A diverged perturbed
    system counts as flipped (clean class destroyed)."""
    if target_fro <= 0:
        with torch.no_grad():
            lg = model.head(Z_sub)[v]
        return (int(lg.argmax()) != p), float(lg[p] - lg[c]), 0.0, 0.0
    dA = _build_symmetric_dA(op, w_dir, target_fro)
    true_fro = float(dA.norm())                          # MEASURED Frobenius norm
    A_pert = (A_sub + dA).contiguous()
    ctx_p = {"A_hat": A_pert, "X_proj": ctx_sub["X_proj"]}
    Z_p, diverged, res = _safe_reconverge(model, Z_sub, ctx_p, max_iter=300, tol=1e-9)
    if diverged:
        return True, float("nan"), true_fro, float("inf")
    with torch.no_grad():
        lg = model.head(Z_p)[v]
    return (int(lg.argmax()) != p), float(lg[p] - lg[c]), true_fro, res


def _empirical_flip_radius(model, A_sub, ctx_sub, Z_sub, op, w_dir,
                           v, p, c, W, lo, hi, iters):
    """Smallest ||dA||_F along w_dir that flips node v (bisection). None if no
    flip even at hi."""
    flip_hi, _, _, _ = _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, hi, v, p, c, W)
    if not flip_hi:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        flip_mid, _, _, _ = _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, mid, v, p, c, W)
        if flip_mid:
            hi = mid
        else:
            lo = mid
    return hi


def run_soundness(model, A_sub, ctx_sub, Z_sub, dense, op, Sc, kappa, cand, tag,
                  sound_nodes, frac_below, frac_above, n_random, writer, log):
    """Worst-case + random soundness gate, IDENTICAL protocol to the pilot, run
    on the rho_v of the CURRENT candidate. Counts breaches strictly below rho_v
    and reports the worst measured ||delta-Ahat||_F + reconverge residuals."""
    dev = Z_sub.device
    d = dense["d"]
    W = model.head.weight.detach()
    preds = dense["preds"]; rho_v = dense["rho_v"]; bind_c = dense["bind_c"]
    correct = dense["correct"]
    E = Sc.shape[1]

    cand_nodes = [v for v in range(A_sub.shape[0])
                  if bool(correct[v]) and float(rho_v[v]) > 1e-4 and int(bind_c[v]) >= 0]
    cand_nodes.sort(key=lambda v: float(rho_v[v]))       # hardest (smallest rho) first
    chosen = cand_nodes[:sound_nodes]

    n_breach_below = 0; n_flip_above = 0
    emp_ratios = []
    max_res_below = 0.0                                   # worst finite reconverge residual at the breach-test radius
    base_pred = preds.clone()

    for v in chosen:
        p = int(base_pred[v]); c = int(bind_c[v]); rho = float(rho_v[v])
        Sv = Sc[v * d:(v + 1) * d, :]
        wgap = (W[p] - W[c])
        g = (wgap @ Sv)                                  # (|E|,) worst-case edge direction
        gn = float(g.norm())
        if gn < 1e-30:
            continue
        w_dir = -(g / gn)

        flipped_below, _, fro_b, res_b = _attack_and_check(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, frac_below * rho, v, p, c, W)
        if math.isfinite(res_b):
            max_res_below = max(max_res_below, res_b)
        if flipped_below:
            n_breach_below += 1

        flipped_above, _, _, _ = _attack_and_check(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, frac_above * rho, v, p, c, W)
        if flipped_above:
            n_flip_above += 1

        A_fro = float(A_sub.norm())
        hi = min(max(12.0 * rho, frac_above * rho), 0.5 * A_fro)
        hi = max(hi, frac_above * rho)
        emp_r = _empirical_flip_radius(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, v, p, c, W, lo=0.0, hi=hi, iters=16)
        if emp_r is None:
            emp_ratio = float("inf"); emp_capped = hi / (rho + 1e-30)
        else:
            emp_ratio = emp_r / (rho + 1e-30); emp_capped = emp_ratio
        emp_ratios.append(emp_capped)

        rand_breaches = 0
        for s in range(n_random):
            torch.manual_seed(10_000 * v + s)
            wr = torch.randn(E, device=dev, dtype=Sc.dtype)
            wr = wr / (wr.norm() + 1e-30)
            fr, _, _, _ = _attack_and_check(
                model, A_sub, ctx_sub, Z_sub, op, wr, frac_below * rho, v, p, c, W)
            if fr:
                rand_breaches += 1; n_breach_below += 1

        writer.writerow({
            "candidate": cand, "tag": tag, "node": v, "pred": p, "binding_c": c,
            "rho_v": rho, "true_fro_below": fro_b,
            "worstcase_flip_below": int(flipped_below),
            "worstcase_flip_above": int(flipped_above),
            "rand_breaches_below": rand_breaches, "n_random": n_random,
            "emp_flip_radius": (emp_r if emp_r is not None else float("nan")),
            "emp_ratio": emp_ratio, "reconv_res_below": res_b,
        })

    finite_ratios = sorted(r for r in emp_ratios if math.isfinite(r))
    med_ratio = finite_ratios[len(finite_ratios) // 2] if finite_ratios else float("inf")
    summary = {
        "candidate": cand, "tag": tag, "n_certified_attacked": len(chosen),
        "n_breach_below_rho": n_breach_below, "n_flip_above": n_flip_above,
        "median_emp_flip_over_rho": med_ratio,
        "min_emp_ratio": (finite_ratios[0] if finite_ratios else float("inf")),
        "max_reconv_res_below": max_res_below,
    }
    log.append(summary)
    return summary


# ---------------------------------------------------------------------------
# Non-vacuity aggregation
# ---------------------------------------------------------------------------
def _frac_above(vals, eps):
    if not vals:
        return float("nan")
    return sum(1 for x in vals if x > eps) / len(vals)


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[len(s) // 2]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,137,271")
    ap.add_argument("--datasets", default="Cora,Citeseer,WikiCS")
    ap.add_argument("--dense-dataset", default="Cora")
    ap.add_argument("--dense-nodes", type=int, default=80)
    ap.add_argument("--fullgraph-sample", type=int, default=300)
    ap.add_argument("--sound-nodes", type=int, default=24)
    ap.add_argument("--frac-below", type=float, default=0.99)
    ap.add_argument("--frac-above", type=float, default=1.5)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--candidates", default="T1,T2,T3")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = args.seeds.split(",")[0]
        args.fullgraph_sample = min(args.fullgraph_sample, 120)
        args.sound_nodes = min(args.sound_nodes, 10)

    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    datasets = [s for s in args.datasets.split(",") if s != ""]
    candidates = [s for s in args.candidates.split(",") if s != ""]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    dense_csv = open(RESULTS / "certify_tighten_dense.csv", "w", newline="")
    dw = csv.DictWriter(dense_csv, fieldnames=[
        "candidate", "tag", "node", "pred", "binding_c", "margin", "rho_v",
        "r_v_linear", "ratio_rho_over_r", "kappa", "L_J_node", "n_nodes", "n_edges"])
    dw.writeheader()

    full_csv = open(RESULTS / "certify_tighten_fullgraph.csv", "w", newline="")
    fw = csv.DictWriter(full_csv, fieldnames=[
        "candidate", "tag", "node", "pred", "binding_c", "margin", "rho_v",
        "r_v_linear", "ratio_rho_over_r", "correct", "kappa", "L_J_node",
        "n_nodes", "n_edges"])
    fw.writeheader()

    snd_csv = open(RESULTS / "certify_tighten_soundness.csv", "w", newline="")
    sw = csv.DictWriter(snd_csv, fieldnames=[
        "candidate", "tag", "node", "pred", "binding_c", "rho_v", "true_fro_below",
        "worstcase_flip_below", "worstcase_flip_above", "rand_breaches_below",
        "n_random", "emp_flip_radius", "emp_ratio", "reconv_res_below"])
    sw.writeheader()

    soundness_summaries = []
    # accumulators: full-graph non-vacuity per (candidate, dataset) across seeds
    full_acc = {}     # (cand, ds) -> {"rho":[], "ratio":[]}
    dense_acc = {}    # (cand,) -> {"rho":[], "ratio":[]}

    # ============ DENSE + SOUNDNESS (dense-dataset, all seeds, all candidates) ============
    for seed in seeds:
        X, A, y, tm, nf, nc = load_dataset(args.dense_dataset)
        X, A, y = X.to(dev), A.to(dev), y.to(dev)
        model = train_ignn(X, A, y, tm, nf, nc, dev, seed=seed, epochs=args.epochs)
        with torch.no_grad():
            _, Zf, ctxf = model(X, A)
        idx = extract_ego_subgraph(A, max_nodes=args.dense_nodes)
        A_sub = A[idx][:, idx].contiguous()
        ctx_sub = {"A_hat": A_sub, "X_proj": ctxf["X_proj"][idx].contiguous()}
        Z_sub = reconverge(model, Zf[idx].clone(), ctx_sub, max_iter=600, tol=1e-9)
        y_sub = y[idx]

        # shared across candidates: equilibrium residual, kappa, Sc, op, edge_list
        opk = ScalableSensitivity(F_op_factory(model), Z_sub, ctx_sub)
        kappa = rho_rayleigh(opk)
        with torch.no_grad():
            res_eq = float((model.operator(Z_sub, ctx_sub) - Z_sub).norm())
        Fop = F_op_factory(model)
        S = structural_sensitivity_matrix(Fop, Z_sub, ctx_sub, "A_hat")
        Sc_code, edge_list = constrained_sensitivity_matrix(S, A_sub)
        Sc = Sc_code / SQRT2
        Nsub = A_sub.shape[0]
        nrm = node_norms(Z_sub)
        print(f"[{args.dense_dataset}/dense/s{seed}] N={Nsub} E={opk.num_edges} "
              f"kappa={kappa:.4f} eqres={res_eq:.1e} "
              f"||z*||_F={float(Z_sub.norm()):.3f} max||z*_i||={float(nrm.max()):.3f} "
              f"ratio_F/max={float(Z_sub.norm())/float(nrm.max()):.2f} "
              f"sqrtN={math.sqrt(Nsub):.2f}", flush=True)

        for cand in candidates:
            desc, LJ_fn = make_LJ_provider(cand, model, Z_sub, edge_list, Nsub)
            tag = f"{args.dense_dataset}/dense/s{seed}"
            dense = certify_dense(model, A_sub, ctx_sub, Z_sub, y_sub, kappa, cand,
                                  tag, LJ_fn, desc, Sc, edge_list, dw)
            dense_csv.flush()
            # accumulate dense non-vacuity (correct nodes only)
            acc = dense_acc.setdefault(cand, {"rho": [], "ratio": []})
            for v in range(Nsub):
                if not bool(dense["correct"][v]):
                    continue
                rho = float(dense["rho_v"][v]); rr = float(dense["r_v"][v])
                acc["rho"].append(rho)
                acc["ratio"].append(rho / (rr + 1e-30))

            ssum = run_soundness(
                model, A_sub, ctx_sub, Z_sub, dense, opk, Sc, kappa, cand, tag,
                sound_nodes=args.sound_nodes, frac_below=args.frac_below,
                frac_above=args.frac_above, n_random=args.n_random,
                writer=sw, log=soundness_summaries)
            snd_csv.flush()
            print(f"  [{cand}] {desc}", flush=True)
            print(f"     attacked={ssum['n_certified_attacked']} "
                  f"BREACH_below_rho={ssum['n_breach_below_rho']} "
                  f"flip@{args.frac_above}rho={ssum['n_flip_above']} "
                  f"med(emp/rho)={ssum['median_emp_flip_over_rho']:.3f} "
                  f"min(emp/rho)={ssum['min_emp_ratio']:.3f} "
                  f"maxreconv_res={ssum['max_reconv_res_below']:.1e}", flush=True)

    # ============ FULL-GRAPH (all datasets, all seeds, all candidates) ============
    for ds in datasets:
        for seed in seeds:
            X, A, y, tm, nf, nc = load_dataset(ds)
            X, A, y = X.to(dev), A.to(dev), y.to(dev)
            model = train_ignn(X, A, y, tm, nf, nc, dev, seed=seed, epochs=args.epochs)
            with torch.no_grad():
                _, Z_star, ctx = model(X, A)
            op = ScalableSensitivity(F_op_factory(model), Z_star, ctx)
            kappa = rho_rayleigh(op)
            if kappa >= 0.98:
                op = ScalableSensitivity(F_op_factory(model), Z_star, ctx,
                                         neumann_terms=3000)
            with torch.no_grad():
                preds = model.head(Z_star).argmax(dim=1)
            acc_cls = float((preds.cpu() == y.cpu()).float().mean())
            N = X.shape[0]
            torch.manual_seed(seed)
            corr_idx = (preds.cpu() == y.cpu()).nonzero(as_tuple=True)[0]
            if corr_idx.numel() > args.fullgraph_sample:
                sel = corr_idx[torch.randperm(corr_idx.numel())[:args.fullgraph_sample]]
            else:
                sel = corr_idx
            sel_list = sel.tolist()
            nrm = node_norms(Z_star)
            edge_pairs = op.edge_list  # existing undirected edges
            print(f"[{ds}/full/s{seed}] N={N} E={op.num_edges} kappa={kappa:.4f} "
                  f"acc={acc_cls:.3f} sampled={len(sel_list)} "
                  f"||z*||_F={float(Z_star.norm()):.3f} max||z*_i||={float(nrm.max()):.3f} "
                  f"ratio_F/max={float(Z_star.norm())/float(nrm.max()):.2f} "
                  f"sqrtN={math.sqrt(N):.2f}", flush=True)

            # T3 needs a 2-hop max over the FULL sparse graph but only for sampled
            # nodes; precompute per-sampled-node to avoid O(N) python loop.
            t3_local = None
            if "T3" in candidates:
                t3_local = _two_hop_for_nodes(nrm, edge_pairs, N, sel_list,
                                              w_norm_2(model) ** 2)

            for cand in candidates:
                if cand == "T3":
                    LJ_fn = lambda v, _m=t3_local: float(_m[int(v)])
                    desc = "L_{J,v} = ||W||^2 * max_{u in 2-hop(v)} ||z*_u||_2"
                else:
                    desc, LJ_fn = make_LJ_provider(cand, model, Z_star, edge_pairs, N)
                tag = f"{ds}/full/s{seed}"
                acc = full_acc.setdefault((cand, ds), {"rho": [], "ratio": []})
                certify_fullgraph(model, X, y, kappa, op, Z_star, cand, tag,
                                  LJ_fn, sel_list, fw, acc=acc)
                full_csv.flush()

    dense_csv.close(); full_csv.close(); snd_csv.close()

    # ============ DELIVERABLE SUMMARY TABLE ============
    sum_csv = open(RESULTS / "certify_tighten_summary.csv", "w", newline="")
    mw = csv.DictWriter(sum_csv, fieldnames=[
        "candidate", "scope", "dataset", "n", "median_rho", "median_ratio",
        "frac_gt_0.01", "frac_gt_0.05", "frac_gt_0.10", "breaches"])
    mw.writeheader()

    # breaches per candidate (summed over all dense seeds)
    breaches_by_cand = {}
    for s in soundness_summaries:
        breaches_by_cand[s["candidate"]] = breaches_by_cand.get(s["candidate"], 0) + s["n_breach_below_rho"]

    print("\n================ SUMMARY ================")
    for cand in candidates:
        # dense row
        da = dense_acc.get(cand, {"rho": [], "ratio": []})
        br = breaches_by_cand.get(cand, 0)
        row = {
            "candidate": cand, "scope": "dense", "dataset": args.dense_dataset,
            "n": len(da["rho"]), "median_rho": _median(da["rho"]),
            "median_ratio": _median(da["ratio"]),
            "frac_gt_0.01": _frac_above(da["rho"], 0.01),
            "frac_gt_0.05": _frac_above(da["rho"], 0.05),
            "frac_gt_0.10": _frac_above(da["rho"], 0.10),
            "breaches": br,
        }
        mw.writerow(row)
        print(f"[{cand}] DENSE {args.dense_dataset}: n={row['n']} "
              f"med_rho={row['median_rho']:.4f} med_ratio={row['median_ratio']:.4f} "
              f"f>0.01={row['frac_gt_0.01']:.3f} f>0.05={row['frac_gt_0.05']:.3f} "
              f"f>0.10={row['frac_gt_0.10']:.3f} BREACHES={br}")
        for ds in datasets:
            fa = full_acc.get((cand, ds), {"rho": [], "ratio": []})
            row = {
                "candidate": cand, "scope": "full", "dataset": ds,
                "n": len(fa["rho"]), "median_rho": _median(fa["rho"]),
                "median_ratio": _median(fa["ratio"]),
                "frac_gt_0.01": _frac_above(fa["rho"], 0.01),
                "frac_gt_0.05": _frac_above(fa["rho"], 0.05),
                "frac_gt_0.10": _frac_above(fa["rho"], 0.10),
                "breaches": br,  # soundness gate is on dense; same constant applies
            }
            mw.writerow(row)
            print(f"[{cand}] FULL  {ds}: n={row['n']} "
                  f"med_rho={row['median_rho']:.4f} med_ratio={row['median_ratio']:.4f} "
                  f"f>0.01={row['frac_gt_0.01']:.3f} f>0.05={row['frac_gt_0.05']:.3f} "
                  f"f>0.10={row['frac_gt_0.10']:.3f}")
    sum_csv.close()

    print("\nSoundness summaries (per dense seed):")
    for s in soundness_summaries:
        print(" ", s)
    print(f"\nDONE in {time.time()-t0:.1f}s")
    print("Wrote results/certify_tighten_{dense,fullgraph,soundness,summary}.csv")


# ---------------------------------------------------------------------------
# Helpers used by the driver (kept after main for readability)
# ---------------------------------------------------------------------------
def _two_hop_for_nodes(node_norm_vec, edge_pairs, N, nodes, w2):
    """Per-node closed-2-hop max embedding norm * w2, only for `nodes` (full
    graph). Builds an adjacency list once; BFS depth 2 per target node."""
    rows = [[] for _ in range(N)]
    for a, b in edge_pairs:
        rows[a].append(b); rows[b].append(a)
    out = {}
    nn_ = node_norm_vec
    for v in nodes:
        v = int(v)
        hop = {v}
        for u in rows[v]:
            hop.add(u)
            for w in rows[u]:
                hop.add(w)
        idx = torch.as_tensor(sorted(hop), device=nn_.device, dtype=torch.long)
        out[v] = w2 * float(nn_[idx].max())
    return out


if __name__ == "__main__":
    main()
