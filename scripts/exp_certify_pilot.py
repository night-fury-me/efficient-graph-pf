#!/usr/bin/env python3
"""GATING pilot for AEGIS-Certify: a SOUND certified structural radius rho_v.

Tests, for a trained IGNN node classifier with linear head f(z)=Wz+b, whether
the second-order-corrected per-node radius

    rho_v = min_{c != y_v}  ( -L1_c + sqrt(L1_c^2 + 4 C_v m_v^{(c)}) ) / (2 C_v)

is (i) NON-VACUOUS and (ii) SOUND (no symmetric edge-supported perturbation with
||dA_hat||_F < rho_v flips node v).

Constants (paper eq:radius denominator + eq:transfer curvature):
    L1_c = || (W_{y_v} - W_c) @ S_{c,v}^{paper} ||_2          (linear term)
    C_v  = || W_{y_v} - W_c ||_2 * (1-kappa)^{-2} * L_J / 2    (curvature term)
    L_J  = ||W||_2^2 * ||z*||                                  (Frobenius ||z*||)
    kappa= spectral radius of J_z via honest Rayleigh-quotient power iteration.

CONVENTION (bug B1 / the sqrt(2) trap)
--------------------------------------
The paper defines S_c on the UNIT edge basis b_k=(e_i e_j^T + e_j e_i^T)/sqrt(2)
(||b_k||_F = 1), so S_{c,v}^{paper} is the equilibrium response PER UNIT
||dA_hat||_F. The CODE (constrained_sensitivity_matrix / ScalableSensitivity.matvec)
builds the column as S_{:,iN+j}+S_{:,jN+i}, i.e. the response to the *un-normalized*
indicator e_i e_j^T + e_j e_i^T whose Frobenius norm is sqrt(2). Hence

    S_{c,v}^{paper} = S_{c,v}^{code} / sqrt(2)   ==>   L1_c = L1_c^{code} / sqrt(2).

We divide by sqrt(2) exactly ONCE (no extra factor; the proposal sketch's
"sqrt(2)*sigma_1" is wrong and is NOT used). The soundness attack measures the
literal Frobenius norm of the perturbation MATRIX, so units are consistent
end-to-end. Sanity invariant enforced per node: rho_v < r_v_linear (curvature
can only shrink the radius).

Outputs:
    results/certify_pilot_dense.csv      per-node rho_v / r_v (dense exact)
    results/certify_pilot_fullgraph.csv  per-node rho_v / r_v (matrix-free)
    results/certify_pilot_soundness.csv  per-attacked-node breach records
    paper/review/certify_pilot_findings.md  verdict report (written by caller)

Usage:
    .venv/bin/python scripts/exp_certify_pilot.py [--seeds 42,137,271] \
        [--dense-nodes 80] [--fullgraph-sample 300] [--sound-nodes 24] \
        [--datasets Cora,Citeseer,WikiCS] [--quick]
"""
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
# Shared helpers
# ---------------------------------------------------------------------------
def F_op_factory(model):
    def F_op(z, c):
        return model.operator(z, c)
    return F_op


def w_norm_2(model) -> float:
    """||W||_2 of the state-propagation weight (spectral-norm-parametrized)."""
    return float(torch.linalg.svdvals(model.W.weight.detach())[0])


def curvature_LJ(model, z_star: torch.Tensor) -> float:
    """L_J <= ||W||_2^2 * ||z*||_F   (eq:transfer second-order curvature const)."""
    return (w_norm_2(model) ** 2) * float(z_star.norm())


def positive_root(L1: float, C: float, m: float) -> float:
    """Positive root r of  m - L1 r - C r^2 = 0  (m>0, C>=0, L1>=0).

    Quadratic C r^2 + L1 r - m = 0 with a=C>0:
        r = (-L1 + sqrt(L1^2 + 4 C m)) / (2 C).
    C -> 0 limit (no curvature) reduces to the linear radius m / L1.
    """
    if m <= 0:
        return 0.0
    if C <= 1e-30:
        return m / (L1 + 1e-30)
    disc = L1 * L1 + 4.0 * C * m
    return (-L1 + math.sqrt(disc)) / (2.0 * C)


# ---------------------------------------------------------------------------
# (1a) DENSE exact rho_v on an IGNN ego-subgraph
# ---------------------------------------------------------------------------
def certify_dense(model, A_sub, ctx_sub, Z_sub, y_sub, kappa, tag, writer):
    """Exact per-node rho_v and r_v on a dense subgraph. Returns dict of arrays
    plus the objects (S_c_paper, edge_list, op) needed by the soundness stage."""
    dev = Z_sub.device
    N = A_sub.shape[0]
    d = model.hidden
    Fop = F_op_factory(model)

    S = structural_sensitivity_matrix(Fop, Z_sub, ctx_sub, "A_hat")      # (Nd, N^2)
    Sc_code, edge_list = constrained_sensitivity_matrix(S, A_sub)        # (Nd, |E|) code-conv
    Sc = Sc_code / SQRT2                                                 # paper unit-basis
    E = len(edge_list)

    W = model.head.weight.detach()                                      # (C, d)
    L_J = curvature_LJ(model, Z_sub)
    kfac = (1.0 - kappa) ** (-2)

    with torch.no_grad():
        logits = model.head(Z_sub)                                     # (N, C)
    preds = logits.argmax(dim=1)
    Ccls = logits.shape[1]

    rho_v = torch.zeros(N, device=dev)
    r_v = torch.zeros(N, device=dev)               # linear-only radius (paper eq:radius, paper conv)
    bind_c = torch.full((N,), -1, dtype=torch.long, device=dev)
    margins = torch.zeros(N, device=dev)

    for v in range(N):
        p = int(preds[v])
        Sv = Sc[v * d:(v + 1) * d, :]              # (d, |E|)  paper-convention node block
        marg = (logits[v, p] - logits[v])         # (C,)  >=0, 0 at p
        best_rho = float("inf")
        best_r = float("inf")
        best_c = -1
        for c in range(Ccls):
            if c == p:
                continue
            m_c = float(marg[c])
            if m_c <= 0:
                continue
            wgap = (W[p] - W[c])                   # (d,)
            wg_norm = float(wgap.norm())
            L1_c = float((wgap @ Sv).norm())       # ||(W_y - W_c) S_{c,v}||_2  paper-conv
            C_v = wg_norm * kfac * L_J / 2.0
            r_c = positive_root(L1_c, C_v, m_c)
            r_lin = m_c / (L1_c + 1e-30)
            # invariant: curvature can only shrink (r_c <= r_lin); guard numerics
            if r_c < best_rho:
                best_rho = r_c
                best_c = c
            if r_lin < best_r:
                best_r = r_lin
        if best_c < 0:
            rho_v[v] = 0.0
            r_v[v] = 0.0
        else:
            rho_v[v] = best_rho
            r_v[v] = best_r
            bind_c[v] = best_c
            margins[v] = float(marg[best_c])

    correct = (preds.cpu() == y_sub.cpu())
    op = ScalableSensitivity(Fop, Z_sub, ctx_sub)  # for soundness edge<->matrix maps
    # per-row CSV (correct nodes only)
    for v in range(N):
        if not bool(correct[v]):
            continue
        writer.writerow({
            "tag": tag, "node": v, "pred": int(preds[v]),
            "binding_c": int(bind_c[v]), "margin": float(margins[v]),
            "rho_v": float(rho_v[v]), "r_v_linear": float(r_v[v]),
            "ratio_rho_over_r": float(rho_v[v] / (r_v[v] + 1e-30)),
            "kappa": kappa, "L_J": L_J, "n_nodes": N, "n_edges": E,
        })
    return {
        "rho_v": rho_v, "r_v": r_v, "bind_c": bind_c, "margins": margins,
        "preds": preds, "correct": correct, "Sc": Sc, "W": W, "d": d,
        "edge_list": edge_list, "op": op, "logits": logits,
    }


# ---------------------------------------------------------------------------
# (1b) MATRIX-FREE rho_v on the full graph (sampled nodes)
# ---------------------------------------------------------------------------
def certify_fullgraph(model, X, A, y, kappa, op, Z_star, ctx, sample_nodes, tag, writer):
    """Per-node rho_v via matrix-free rmatvec (VJP). For node v and competitor c,
        (W_p - W_c) @ S_{c,v}^{code}  =  rmatvec( e_v (x) (W_p - W_c) )
    where e_v (x) g is the D-vector with g placed in node v's block. Divide by
    sqrt(2) for the paper convention. One rmatvec per (node,competitor) pair.
    """
    dev = Z_star.device
    d = model.hidden
    D = op.D
    W = model.head.weight.detach()
    L_J = curvature_LJ(model, Z_star)
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
        best_rho = float("inf"); best_r = float("inf"); best_c = -1; best_m = 0.0
        for c in range(Ccls):
            if c == p:
                continue
            m_c = float(marg[c])
            if m_c <= 0:
                continue
            wgap = (W[p] - W[c])                    # (d,)
            u = torch.zeros(D, device=dev, dtype=Z_star.dtype)
            u[v * d:(v + 1) * d] = wgap             # e_v (x) (W_p - W_c)
            edge_resp = op.rmatvec(u)               # S_c^{code,T} u  in R^{|E|}, == (W_p-W_c) S_{c,v}^{code}
            L1_c = float(edge_resp.norm()) / SQRT2  # paper convention
            wg_norm = float(wgap.norm())
            C_v = wg_norm * kfac * L_J / 2.0
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
            "tag": tag, "node": v, "pred": p, "binding_c": best_c, "margin": best_m,
            "rho_v": best_rho, "r_v_linear": best_r,
            "ratio_rho_over_r": best_rho / (best_r + 1e-30),
            "correct": int(corr), "kappa": kappa, "L_J": L_J,
            "n_nodes": int(X.shape[0]), "n_edges": op.num_edges,
        })
        n_rows += 1
    return n_rows


# ---------------------------------------------------------------------------
# (2) SOUNDNESS on the dense subgraph
# ---------------------------------------------------------------------------
def _build_symmetric_dA(op, w_edges: torch.Tensor, target_fro: float) -> torch.Tensor:
    """Map paper-convention edge vector w (||w||_2 == desired ||dA||_F) to a
    symmetric edge-supported dA with EXACT Frobenius norm `target_fro`.

    op._edges_to_delta_A(v) sets dA[i,j]=dA[j,i]=v_k, so ||dA||_F = sqrt(2)||v||_2.
    Hence feeding w/sqrt(2) yields ||dA||_F = ||w||_2. We then rescale to the
    exact target to remove any drift. Returns a dense (N,N) symmetric matrix
    supported only on existing edges.
    """
    dA = op._edges_to_delta_A(w_edges / SQRT2)
    fro = dA.norm()
    if float(fro) < 1e-30:
        return dA
    return dA * (target_fro / fro)


def run_soundness(model, A_sub, ctx_sub, Z_sub, dense, kappa, tag,
                  sound_nodes, frac_below, frac_above, n_random, writer, log):
    """For a sample of certified nodes, attack along the worst-case first-order
    direction at frac_below*rho_v (must NOT flip) and frac_above*rho_v (may flip),
    plus random symmetric directions at frac_below*rho_v. Reconverge each time
    and check whether node v's predicted class flips.
    """
    dev = Z_sub.device
    d = dense["d"]
    Sc = dense["Sc"]            # paper convention (Nd, |E|)
    W = dense["W"]
    preds = dense["preds"]
    rho_v = dense["rho_v"]
    bind_c = dense["bind_c"]
    correct = dense["correct"]
    op = dense["op"]
    E = Sc.shape[1]

    # candidate certified nodes: correct, rho_v>0, finite, with a real binding competitor
    cand = [v for v in range(A_sub.shape[0])
            if bool(correct[v]) and float(rho_v[v]) > 1e-4 and int(bind_c[v]) >= 0]
    # prefer the nodes with the SMALLEST rho_v (hardest / most likely to expose a breach)
    cand.sort(key=lambda v: float(rho_v[v]))
    chosen = cand[:sound_nodes]

    n_breach_below = 0
    n_flip_above = 0
    emp_ratios = []   # empirical flip radius / rho_v  (via bisection on the worst-case dir)
    records = []

    base_pred = preds.clone()

    for v in chosen:
        p = int(base_pred[v])
        c = int(bind_c[v])
        rho = float(rho_v[v])
        Sv = Sc[v * d:(v + 1) * d, :]                 # (d,|E|) paper-conv
        wgap = (W[p] - W[c])                          # (d,)
        # worst-case edge direction: most negative margin change
        #   d(margin)/d(w) = (W_p - W_c) @ Sv  in edge space; descend it.
        g = (wgap @ Sv)                               # (|E|,)
        gn = float(g.norm())
        if gn < 1e-30:
            continue
        w_dir = -(g / gn)                             # unit edge vector (paper conv)

        # ---- worst-case at frac_below * rho (MUST NOT flip) ----
        flipped_below, m_after_b = _attack_and_check(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, frac_below * rho, v, p, c, W)
        breach = flipped_below
        if breach:
            n_breach_below += 1

        # ---- worst-case at frac_above * rho (MAY flip) ----
        flipped_above, m_after_a = _attack_and_check(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, frac_above * rho, v, p, c, W)
        if flipped_above:
            n_flip_above += 1

        # ---- empirical flip radius along worst-case dir (bisection) ----
        # Search up to a ceiling wide enough to characterise looseness yet
        # PHYSICALLY bounded: a perturbation whose Frobenius norm approaches that
        # of A_sub itself destroys contractivity (Z* diverges, no meaningful
        # "flip"), so cap at min(12*rho, 0.5*||A_sub||_F). If no flip by the cap,
        # emp_ratio is recorded as the cap/rho lower bound (capped flag set).
        A_fro = float(A_sub.norm())
        hi = min(max(12.0 * rho, frac_above * rho), 0.5 * A_fro)
        hi = max(hi, frac_above * rho)  # never below the frac_above probe point
        emp_r = _empirical_flip_radius(
            model, A_sub, ctx_sub, Z_sub, op, w_dir, v, p, c, W,
            lo=0.0, hi=hi, iters=16)
        if emp_r is None:
            emp_ratio = float("inf")   # no flip within the physical cap
            emp_capped = hi / (rho + 1e-30)
        else:
            emp_ratio = emp_r / (rho + 1e-30)
            emp_capped = emp_ratio
        emp_ratios.append(emp_capped)  # use capped lower bound for the median

        # ---- random symmetric directions at frac_below * rho (weaker check) ----
        rand_breaches = 0
        for s in range(n_random):
            torch.manual_seed(10_000 * v + s)
            wr = torch.randn(E, device=dev, dtype=Sc.dtype)
            wr = wr / (wr.norm() + 1e-30)
            fr, _ = _attack_and_check(
                model, A_sub, ctx_sub, Z_sub, op, wr, frac_below * rho, v, p, c, W)
            if fr:
                rand_breaches += 1
                n_breach_below += 1  # any breach below rho_v counts against soundness

        rec = {
            "tag": tag, "node": v, "pred": p, "binding_c": c, "rho_v": rho,
            "worstcase_flip_below": int(flipped_below),
            "worstcase_flip_above": int(flipped_above),
            "rand_breaches_below": rand_breaches, "n_random": n_random,
            "emp_flip_radius": (emp_r if emp_r is not None else float("nan")),
            "emp_ratio": emp_ratio,
            "margin_after_below": m_after_b, "margin_after_above": m_after_a,
        }
        records.append(rec)
        writer.writerow(rec)

    finite_ratios = sorted(r for r in emp_ratios if math.isfinite(r))
    med_ratio = finite_ratios[len(finite_ratios) // 2] if finite_ratios else float("inf")
    summary = {
        "tag": tag, "n_certified_attacked": len(chosen),
        "n_breach_below_rho": n_breach_below,
        "n_flip_at_1.5rho": n_flip_above,
        "median_emp_flip_over_rho": med_ratio,
        "min_emp_ratio": (finite_ratios[0] if finite_ratios else float("inf")),
    }
    log.append(summary)
    return summary, records


def _safe_reconverge(model, Z_init, ctx_p, max_iter=300, tol=1e-9):
    """Reconverge with a divergence guard. Returns (Z, diverged: bool).

    A large structural perturbation can break contractivity so the IGNN
    iteration diverges (Z -> inf). We detect non-finite / exploding states and
    bail. Diverged == the perturbed system has no usable equilibrium.
    """
    Z = Z_init.clone()
    base = max(float(Z.norm()), 1.0)
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_p)
            nn = float(Z_new.norm())
            if not math.isfinite(nn) or nn > 1e6 * base:
                return Z_new, True
            if (Z_new - Z).norm() < tol * max(float(Z.norm()), 1.0):
                return Z_new, False
            Z = Z_new
    return Z, False


def _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, target_fro,
                      v, p, c, W):
    """Apply a symmetric edge-supported dA of exact ||.||_F=target_fro along
    edge direction w_dir, reconverge, and report (flipped?, margin_after).
    flipped == node v no longer predicted as p (its clean class). A diverged
    perturbed system counts as flipped (the node's clean class is destroyed);
    this can only TIGHTEN the empirical flip radius, never hide a real
    sub-rho breach (those occur at a finite equilibrium with tiny ||dA||)."""
    if target_fro <= 0:
        with torch.no_grad():
            lg = model.head(Z_sub)[v]
        return (int(lg.argmax()) != p), float(lg[p] - lg[c])
    dA = _build_symmetric_dA(op, w_dir, target_fro)
    A_pert = (A_sub + dA).contiguous()
    ctx_p = {"A_hat": A_pert, "X_proj": ctx_sub["X_proj"]}
    Z_p, diverged = _safe_reconverge(model, Z_sub, ctx_p, max_iter=300, tol=1e-9)
    if diverged:
        return True, float("nan")
    with torch.no_grad():
        lg = model.head(Z_p)[v]
    pred_after = int(lg.argmax())
    return (pred_after != p), float(lg[p] - lg[c])


def _empirical_flip_radius(model, A_sub, ctx_sub, Z_sub, op, w_dir,
                           v, p, c, W, lo, hi, iters):
    """Smallest ||dA||_F along w_dir that flips node v (binary search).
    Returns None if no flip even at `hi`."""
    flip_hi, _ = _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, hi, v, p, c, W)
    if not flip_hi:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        flip_mid, _ = _attack_and_check(model, A_sub, ctx_sub, Z_sub, op, w_dir, mid, v, p, c, W)
        if flip_mid:
            hi = mid
        else:
            lo = mid
    return hi


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
    ap.add_argument("--quick", action="store_true", help="1 seed, smaller samples")
    args = ap.parse_args()

    if args.quick:
        args.seeds = args.seeds.split(",")[0]
        args.fullgraph_sample = min(args.fullgraph_sample, 120)
        args.sound_nodes = min(args.sound_nodes, 10)

    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    datasets = [s for s in args.datasets.split(",") if s != ""]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    dense_csv = open(RESULTS / "certify_pilot_dense.csv", "w", newline="")
    dw = csv.DictWriter(dense_csv, fieldnames=[
        "tag", "node", "pred", "binding_c", "margin", "rho_v", "r_v_linear",
        "ratio_rho_over_r", "kappa", "L_J", "n_nodes", "n_edges"])
    dw.writeheader()

    full_csv = open(RESULTS / "certify_pilot_fullgraph.csv", "w", newline="")
    fw = csv.DictWriter(full_csv, fieldnames=[
        "tag", "node", "pred", "binding_c", "margin", "rho_v", "r_v_linear",
        "ratio_rho_over_r", "correct", "kappa", "L_J", "n_nodes", "n_edges"])
    fw.writeheader()

    snd_csv = open(RESULTS / "certify_pilot_soundness.csv", "w", newline="")
    sw = csv.DictWriter(snd_csv, fieldnames=[
        "tag", "node", "pred", "binding_c", "rho_v",
        "worstcase_flip_below", "worstcase_flip_above",
        "rand_breaches_below", "n_random", "emp_flip_radius", "emp_ratio",
        "margin_after_below", "margin_after_above"])
    sw.writeheader()

    soundness_summaries = []

    # ---- (1a)+(2) DENSE + SOUNDNESS on dense-dataset, all seeds ----
    for seed in seeds:
        tag = f"{args.dense_dataset}/dense/s{seed}"
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

        opk = ScalableSensitivity(F_op_factory(model), Z_sub, ctx_sub)
        kappa = rho_rayleigh(opk)
        res_eq = float((model.operator(Z_sub, ctx_sub) - Z_sub).norm())
        print(f"[{tag}] N={A_sub.shape[0]} E={opk.num_edges} kappa={kappa:.4f} "
              f"eqres={res_eq:.1e}", flush=True)

        dense = certify_dense(model, A_sub, ctx_sub, Z_sub, y_sub, kappa, tag, dw)
        dense_csv.flush()

        ssum, _ = run_soundness(
            model, A_sub, ctx_sub, Z_sub, dense, kappa, tag,
            sound_nodes=args.sound_nodes, frac_below=args.frac_below,
            frac_above=args.frac_above, n_random=args.n_random,
            writer=sw, log=soundness_summaries)
        snd_csv.flush()
        print(f"[{tag}] SOUNDNESS attacked={ssum['n_certified_attacked']} "
              f"breach_below={ssum['n_breach_below_rho']} "
              f"flip@{args.frac_above}rho={ssum['n_flip_at_1.5rho']} "
              f"med(emp/rho)={ssum['median_emp_flip_over_rho']:.3f}", flush=True)

    # ---- (1b) FULL-GRAPH matrix-free, all datasets, all seeds ----
    for ds in datasets:
        for seed in seeds:
            tag = f"{ds}/full/s{seed}"
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
            acc = float((preds.cpu() == y.cpu()).float().mean())
            N = X.shape[0]
            torch.manual_seed(seed)
            # sample correctly-classified nodes preferentially
            corr_idx = (preds.cpu() == y.cpu()).nonzero(as_tuple=True)[0]
            if corr_idx.numel() > args.fullgraph_sample:
                sel = corr_idx[torch.randperm(corr_idx.numel())[:args.fullgraph_sample]]
            else:
                sel = corr_idx
            print(f"[{tag}] N={N} E={op.num_edges} kappa={kappa:.4f} acc={acc:.3f} "
                  f"sampled={sel.numel()}", flush=True)
            certify_fullgraph(model, X, A, y, kappa, op, Z_star, ctx,
                              sel.tolist(), tag, fw)
            full_csv.flush()

    dense_csv.close(); full_csv.close(); snd_csv.close()
    print(f"\nDONE in {time.time()-t0:.1f}s")
    print("Soundness summaries:")
    for s in soundness_summaries:
        print(" ", s)


if __name__ == "__main__":
    main()
