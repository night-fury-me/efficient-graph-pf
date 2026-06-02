"""Experiment #5 — Numerical validation of the attack–defense coupling proposition.

PROPOSITION (paper/review/rec_4_5_6_framing.md, #5). With
    S_c = (I - J_z)^{-1} J_A P_c,   leading right singular pair (sigma_1, v_1),
ATTACK and DEFENSE are two readings of the SAME operator:
  (a) coupling.  The per-node sensitivity ||S_{c,v}|| that shrinks the certified
      radius rho_v is the v-block norm of the operator whose top singular value
      sigma_1 is the attack gain. The least-certifiable nodes lie on the support
      of the optimal attack direction v_1.
  (b) shared divergence.  As kappa = rho(J_z) -> 1 the resolvent (I - J_z)^{-1}
      blows up, so sigma_1 -> inf AND rho_v -> 0 simultaneously.

This script tests both numerically by REUSING the repo's verified machinery
(S_c / certify / SVD); it does NOT reinvent any of it.

  PART (a)  multi-seed coupling correlation (cheap, CPU/GPU, reuses trained models)
            Seeds 42,137,271,314,1729,2718,3141,5772,6561,9999. Per seed, on the
            c=0.9 Cora IGNN (train_regularized lam=0):
              * v_1, sigma_1                              (svd_direction)
              * per-node rho_v, margin m_v, correctness   (certify_fullgraph, T3)
              * per-node sensitivity block norm s_v       (matrix-free, see below)
              * per-node attack exposure a_v from |v_1| on node v's edges
                (L1 / L2 / max, SAME edge basis as op.edge_list)
            Spearman correlations (per seed -> mean±std across seeds):
              (i)   a_v vs s_v     CORE coupling, no margin confound  (>0)
              (ii)  s_v vs rho_v   sensitivity shrinks the radius      (<0)
              (iii) a_v vs rho_v   attack support = least certifiable  (<0)
              (iv)  a_v vs rho_v PARTIAL controlling for margin m_v    (<0)
            Plus a PERMUTATION NULL for (iii) (1000 shuffles -> z, empirical p),
            and robustness of (iii) across the 3 a_v definitions.

  PART (b)  kappa-divergence sweep (small GPU, seed 42)
            Train Cora IGNN at c in {0.5,0.7,0.9,0.95,0.99}. Per model measure:
              clean kappa = rho(J_z); RESOLVENT 2-norm ||(I - J_z)^{-1}||_2
              (power iteration on the resolvent via Neumann solve) -- the
              CONFOUND-FREE divergence quantity; sigma_1(S_c); cert_frac (T3,
              sampled); test acc.
            Expect kappa^ -> resolvent^ and sigma_1^ (clean divergence) while
            cert_frac declines (with the honest caveat that cert_frac also moves
            with margin/accuracy). This is the CLEAN-kappa capacity knob, NOT the
            Theorem-1 eps_crit perturbation transition.

USAGE
    .venv/bin/python scripts/exp_coupling_validation.py            # full run
    .venv/bin/python scripts/exp_coupling_validation.py --smoke    # seed 42 only,
                                                                   # small sample
Outputs: results/coupling_validation_partA_*.csv, partB.csv, and the per-seed
correlation summary printed to stdout (captured into the findings md).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# ---- REUSE verified machinery (do NOT reinvent) ---------------------------
from scripts.exp_aegis_regularized_training import (
    load_cora,            # -> (X, A, y, train_mask, test_mask, nfeat, ncls) on device
    train_regularized,    # c=0.9 baseline at lam=0 (revision-R2 recipe)
    test_accuracy,
)
from scripts.exp_fullgraph_attack_table import (
    build_op,             # -> (op, Z_star, ctx, rho, rebuilt); rebuilds at high rho
    svd_direction,        # -> (v1, sigma1); v1 in edge space (op.edge_list basis)
    rho_rayleigh,         # rho(J_z) via Rayleigh-quotient power iteration
)
from scripts.exp_certify_tighten import (
    make_LJ_provider,     # T3 curvature provider
    certify_fullgraph,    # per-node rho_v, margin, correct rows via writerow
)
from iem.examples.ignn_cora import IGNN
from iem.scalable import ScalableSensitivity

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
RESULTS = PROJ_ROOT / "results"


# ===========================================================================
# Rank-correlation helpers (Spearman; NEVER Pearson on these magnitudes).
# ===========================================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties shared), matching scipy's 'average' method."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # resolve ties to the average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0           # average rank per unique value
    return avg[inv]


def spearman(x, y):
    """Spearman rho + two-sided p (t-approx). Falls back to scipy if present."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n
    if _HAVE_SCIPY:
        r, p = spearmanr(x, y)
        return float(r), float(p), n
    rx, ry = _rankdata(x), _rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    # two-sided t-approximation
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # survival of |t| under t_{n-2} via normal approx for large n
        from math import erfc
        p = erfc(abs(t) / math.sqrt(2.0))
    return r, float(p), n


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z: correlate the
    residuals of x~z and y~z in RANK space (rank-based partial correlation)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), n
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)

    def resid(a, b):                      # residual of a after regressing on b
        b1 = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(ex, ey)[0, 1])
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 3) / (1 - r * r))   # one control variable
        from math import erfc
        p = erfc(abs(t) / math.sqrt(2.0))
    return r, float(p), n


def permutation_null(a, rho, n_perm=1000, seed=0):
    """Permutation null for Spearman(a, rho): shuffle a, recompute n_perm times.
    Returns (observed_rho, z, empirical_p_two_sided)."""
    obs, _, _ = spearman(a, rho)
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    rho = np.asarray(rho, float)
    mask = np.isfinite(a) & np.isfinite(rho)
    a, rho = a[mask], rho[mask]
    null = np.empty(n_perm)
    rr = _rankdata(rho)
    for k in range(n_perm):
        ap = rng.permutation(a)
        ra = _rankdata(ap)
        null[k] = np.corrcoef(ra, rr)[0, 1]
    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 0 else float("nan")
    # two-sided empirical p (how often |null| >= |obs|)
    p_emp = float((np.abs(null) >= abs(obs)).mean())
    return obs, z, p_emp


# ===========================================================================
# Per-node sensitivity block norm  s_v = ||S_{c,v}||_2
#
# z* is (N, d) and flattens row-major, so node v's hidden rows are the
# contiguous block [v*d : (v+1)*d] of the length-D vector returned by matvec.
#   M_v : R^|E| -> R^d,   M_v u = (S_c u)[v-block]            (forward)
#   M_v^T r = S_c^T (e_v (x) r)  = op.rmatvec(zeros with block v = r)  (adjoint)
# ||S_{c,v}||_2 = sigma_max(M_v) via power iteration on M_v^T M_v.
# EXACT per the proposition. Done only over the certified-node sample (bounded).
# ===========================================================================
def block_norms_exact(op: ScalableSensitivity, nodes, d: int, N: int,
                      n_iter: int = 8, seed: int = 0):
    """Exact s_v for each node in `nodes` via power iteration. Returns {v: s_v}.
    Reuses op.matvec / op.rmatvec only (matrix-free)."""
    out = {}
    g = torch.Generator(device="cpu")
    for v in nodes:
        v = int(v)
        torch.manual_seed(seed + v)
        u = torch.randn(op.num_edges, device=op.device, dtype=op.dtype)
        nu = u.norm()
        if nu < 1e-30:
            out[v] = 0.0
            continue
        u = u / nu
        sig = 0.0
        for _ in range(n_iter):
            Su = op.matvec(u)                      # (D,)
            blk = Su.reshape(N, d)[v]              # (d,)  node v's response
            # adjoint: embed blk back into a full-D vector with only block v set
            full = torch.zeros(N, d, device=op.device, dtype=op.dtype)
            full[v] = blk
            w = op.rmatvec(full.reshape(-1))        # (|E|,)  = M_v^T M_v u
            nw = w.norm()
            if nw < 1e-30:
                sig = 0.0
                break
            sig = math.sqrt(float(nw.item()))      # ||M_v^T M_v u|| -> sigma^2
            u = w / nw
        out[v] = sig
    return out


def block_norms_proxy(Sc_v1: torch.Tensor, d: int, N: int):
    """Principled PROXY for the per-node sensitivity: the per-node magnitude of
    the worst-case (sigma_1) response, ||(S_c v_1)_v||_2. This is exactly the
    response felt at node v under the optimal attack direction, so it is the
    sensitivity channel the attack excites. Returns np.ndarray (N,)."""
    resp = Sc_v1.reshape(N, d)
    return resp.norm(dim=1).detach().cpu().numpy()


# ===========================================================================
# Attack exposure a_v from v_1 in op.edge_list basis (L1 / L2 / max).
# v_1[e] is the weight on undirected edge e=(i,j); node v's exposure aggregates
# |v_1[e]| over edges incident to v. SAME basis/order as op.edge_list.
# ===========================================================================
def attack_exposure(v1: torch.Tensor, edge_list, N: int):
    v1a = v1.detach().abs().cpu().numpy()
    aL1 = np.zeros(N); aL2 = np.zeros(N); aMax = np.zeros(N)
    for e, (i, j) in enumerate(edge_list):
        w = float(v1a[e])
        aL1[i] += w;      aL1[j] += w
        aL2[i] += w * w;  aL2[j] += w * w
        if w > aMax[i]:   aMax[i] = w
        if w > aMax[j]:   aMax[j] = w
    return {"L1": aL1, "L2": aL2, "max": aMax}


# ===========================================================================
# PART (a): per-seed coupling on the c=0.9 baseline.
# ===========================================================================
def run_part_a_seed(seed, device, epochs, cert_sample, n_block_iter, log,
                    block_sample=300):
    X, A, y, train_mask, test_mask, nfeat, ncls = load_cora(device)
    model = train_regularized(
        X, A, y, train_mask, nfeat, ncls, device, seed,
        lam=0.0, epochs=epochs, k_neumann=30, n_power=4, penalty_every=10,
        penalty_form="raw", log=lambda *a, **k: None,
    )
    acc = test_accuracy(model, X, A, y, test_mask)

    op, Z_star, ctx, rho, rebuilt = build_op(model, X, A)
    v1, sigma1 = svd_direction(op)
    edge_list = op.edge_list
    N = X.shape[0]
    d = model.hidden

    exposure = attack_exposure(v1, edge_list, N)        # dict L1/L2/max -> (N,)

    # --- per-node certified radius / margin / correctness (T3), reuse verbatim
    desc, LJ_fn = make_LJ_provider("T3", model, Z_star, edge_list, N)
    with torch.no_grad():
        preds = model.head(Z_star).argmax(dim=1)
    corr_idx = (preds == y).nonzero(as_tuple=True)[0]
    torch.manual_seed(seed)
    if cert_sample and corr_idx.numel() > cert_sample:
        sel = corr_idx[torch.randperm(corr_idx.numel(), device=corr_idx.device)[:cert_sample]]
    else:
        sel = corr_idx
    sel_list = sorted(int(v) for v in sel.tolist())

    rows = []

    class _W:
        def writerow(self, row):
            rows.append(row)

    certify_fullgraph(model, X, y, rho, op, Z_star, "T3", f"s{seed}",
                      LJ_fn, sel_list, _W())

    # collect per-node arrays aligned to certified rows (correct + margin>0 only)
    node_idx, rho_v, margin_v, correct_v = [], [], [], []
    for r in rows:
        node_idx.append(int(r["node"]))
        rho_v.append(float(r["rho_v"]))
        margin_v.append(float(r["margin"]))
        correct_v.append(int(r["correct"]))
    node_idx = np.array(node_idx)
    rho_v = np.array(rho_v)
    margin_v = np.array(margin_v)
    correct_v = np.array(correct_v)
    # keep correctly-classified, finite-radius nodes (cert is defined there)
    keep = (correct_v == 1) & np.isfinite(rho_v)
    node_idx, rho_v, margin_v = node_idx[keep], rho_v[keep], margin_v[keep]

    # --- per-node sensitivity s_v
    #   proxy: ||(S_c v_1)_v||  (cheap, ALL certified nodes). This is the per-node
    #   magnitude of the worst-case (sigma_1) response -- the sensitivity channel
    #   the attack actually excites.
    Sc_v1 = op.matvec(v1)
    s_proxy_all = block_norms_proxy(Sc_v1, d, N)          # (N,)
    s_proxy = s_proxy_all[node_idx]
    #   exact: ||S_{c,v}||_2 (omnidirectional operator block norm) via matrix-free
    #   power iteration. Each node costs ~2*n_iter Neumann solves, so restrict to a
    #   bounded random subsample of the certified set (>= a few hundred is ample
    #   for a Spearman). Indices of that subsample within node_idx are bsub_pos.
    n_cert = len(node_idx)
    rng = np.random.default_rng(seed)
    if block_sample and n_cert > block_sample:
        bsub_pos = np.sort(rng.choice(n_cert, size=block_sample, replace=False))
    else:
        bsub_pos = np.arange(n_cert)
    bsub_nodes = node_idx[bsub_pos].tolist()
    bn = block_norms_exact(op, bsub_nodes, d, N, n_iter=n_block_iter, seed=seed)
    s_exact = np.array([bn[int(v)] for v in bsub_nodes])   # aligned to bsub_pos
    rho_sub = rho_v[bsub_pos]
    margin_sub = margin_v[bsub_pos]

    # exposure restricted to the certified node set (and the exact subsample)
    aL1 = exposure["L1"][node_idx]
    aL2 = exposure["L2"][node_idx]
    aMx = exposure["max"][node_idx]
    aL1_sub = aL1[bsub_pos]; aL2_sub = aL2[bsub_pos]; aMx_sub = aMx[bsub_pos]

    # per-node s_exact aligned back to the full certified set (NaN where not sampled)
    s_exact_full = np.full(n_cert, np.nan)
    s_exact_full[bsub_pos] = s_exact

    out = {
        "seed": seed, "acc": acc, "sigma1": sigma1, "kappa": rho,
        "rebuilt": int(rebuilt), "n_cert": int(n_cert),
        "n_block": int(len(bsub_nodes)),
        "node_idx": node_idx, "rho_v": rho_v, "margin_v": margin_v,
        "s_exact": s_exact_full, "s_proxy": s_proxy,
        "aL1": aL1, "aL2": aL2, "aMax": aMx,
    }

    # --- correlations for this seed -------------------------------------------
    cors = {}
    # (i) a_v vs s_v  (exact on subsample + proxy on all; L1 primary, all 3 robust)
    for tag, a_sub, a_all in (("L1", aL1_sub, aL1), ("L2", aL2_sub, aL2),
                              ("max", aMx_sub, aMx)):
        cors[f"(i)a{tag}_vs_sexact"] = spearman(a_sub, s_exact)
        cors[f"(i)a{tag}_vs_sproxy"] = spearman(a_all, s_proxy)
    # (ii) s_v vs rho_v  (exact on subsample, proxy on all)
    cors["(ii)sexact_vs_rho"] = spearman(s_exact, rho_sub)
    cors["(ii)sproxy_vs_rho"] = spearman(s_proxy, rho_v)
    # (iii) a_v vs rho_v  (all 3 exposures)
    for tag, a in (("L1", aL1), ("L2", aL2), ("max", aMx)):
        cors[f"(iii)a{tag}_vs_rho"] = spearman(a, rho_v)
    # (iv) a_v vs rho_v PARTIAL controlling for margin m_v (all 3 exposures)
    for tag, a in (("L1", aL1), ("L2", aL2), ("max", aMx)):
        cors[f"(iv)a{tag}_vs_rho|margin"] = partial_spearman(a, rho_v, margin_v)
    # permutation null for (iii) with L1 exposure
    obs, z, p_emp = permutation_null(aL1, rho_v, n_perm=1000, seed=seed)
    cors["(iii_perm)aL1_vs_rho"] = (obs, z, p_emp)

    out["cors"] = cors
    log(f"  seed {seed}: acc={acc:.3f} kappa={rho:.4f} sigma1={sigma1:.3f} "
        f"n_cert={len(node_idx)} n_block={out['n_block']} rebuilt={rebuilt}")
    log(f"     (i) aL1~s_exact rho={cors['(i)aL1_vs_sexact'][0]:+.3f} "
        f"(ii) s_exact~rho rho={cors['(ii)sexact_vs_rho'][0]:+.3f} "
        f"(iii) aL1~rho rho={cors['(iii)aL1_vs_rho'][0]:+.3f} "
        f"(iv) aL1~rho|m rho={cors['(iv)aL1_vs_rho|margin'][0]:+.3f} "
        f"perm z={z:+.1f} p={p_emp:.3f}")
    del op
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def aggregate_part_a(per_seed, log):
    """Mean±std across seeds for each correlation key; write a CSV row per key."""
    keys = list(per_seed[0]["cors"].keys())
    summary = {}
    for k in keys:
        coefs = np.array([s["cors"][k][0] for s in per_seed], dtype=float)
        ps = np.array([s["cors"][k][1] for s in per_seed], dtype=float)
        coefs_f = coefs[np.isfinite(coefs)]
        summary[k] = {
            "mean": float(np.mean(coefs_f)) if len(coefs_f) else float("nan"),
            "std": float(np.std(coefs_f)) if len(coefs_f) else float("nan"),
            "min": float(np.min(coefs_f)) if len(coefs_f) else float("nan"),
            "max": float(np.max(coefs_f)) if len(coefs_f) else float("nan"),
            "p_med": float(np.nanmedian(ps)) if len(ps) else float("nan"),
            "n_seed": int(len(coefs_f)),
        }
    return summary


# ===========================================================================
# PART (b): kappa-divergence sweep at varying c (seed 42).
#
# RESOLVENT 2-norm ||(I - J_z)^{-1}||_2 via power iteration on the resolvent.
# Apply R = (I - J_z)^{-1} with op._neumann_solve, and R^T with the adjoint
# solve; power-iterate R^T R. This is the CONFOUND-FREE divergence quantity:
# it depends only on J_z, not on margins/accuracy.
# ===========================================================================
def resolvent_opnorm(op: ScalableSensitivity, iters: int = 60, seed: int = 0) -> float:
    """||(I - J_z)^{-1}||_2 via power iteration on R^T R, R = neumann solve."""
    torch.manual_seed(seed)
    v = torch.randn(op.D, device=op.device, dtype=op.dtype)
    v = v / v.norm()
    sig = 0.0
    for _ in range(iters):
        Rv = op._neumann_solve(v)                 # (I - J_z)^{-1} v
        RtRv = op._neumann_solve_adjoint(Rv)      # (I - J_z)^{-T} (I - J_z)^{-1} v
        nu = RtRv.norm()
        if nu < 1e-30:
            return 0.0
        sig = math.sqrt(float(nu.item()))
        v = RtRv / nu
    return sig


def cert_frac_sampled(model, X, y, op, Z_star, kappa, cert_sample, seed):
    """Fraction of correct nodes with sound rho_v > 0.05 (T3), on a sample."""
    if kappa >= 1.0:
        return 0.0, 0, 0
    N = X.shape[0]
    desc, LJ_fn = make_LJ_provider("T3", model, Z_star, op.edge_list, N)
    with torch.no_grad():
        preds = model.head(Z_star).argmax(dim=1)
    corr_idx = (preds == y).nonzero(as_tuple=True)[0]
    torch.manual_seed(seed)
    if cert_sample and corr_idx.numel() > cert_sample:
        sel = corr_idx[torch.randperm(corr_idx.numel(), device=corr_idx.device)[:cert_sample]]
    else:
        sel = corr_idx
    sel_list = sorted(int(v) for v in sel.tolist())
    rows = []

    class _W:
        def writerow(self, row):
            rows.append(row)

    certify_fullgraph(model, X, y, kappa, op, Z_star, "T3", "kb", LJ_fn, sel_list, _W())
    ncorr = sum(1 for r in rows if int(r["correct"]) == 1)
    ncert = sum(1 for r in rows if int(r["correct"]) == 1 and float(r["rho_v"]) > 0.05)
    frac = ncert / ncorr if ncorr else 0.0
    return frac, ncert, ncorr


def run_part_b(device, c_values, epochs, cert_sample, log):
    seed = 42
    X, A, y, train_mask, test_mask, nfeat, ncls = load_cora(device)
    out_rows = []
    for c in c_values:
        t0 = time.time()
        # train_regularized fixes c=0.9 in its signature default; build IGNN here
        # with the desired c and run the SAME recipe inline is over-engineering --
        # instead call train_regularized with the c override it already supports.
        model = train_regularized(
            X, A, y, train_mask, nfeat, ncls, device, seed,
            lam=0.0, epochs=epochs, k_neumann=30, n_power=4, penalty_every=10,
            c=c, penalty_form="raw", log=lambda *a, **k: None,
        )
        acc = test_accuracy(model, X, A, y, test_mask)
        op, Z_star, ctx, rho, rebuilt = build_op(model, X, A)
        v1, sigma1 = svd_direction(op)
        res_norm = resolvent_opnorm(op, iters=60, seed=0)
        cfrac, ncert, ncorr = cert_frac_sampled(model, X, y, op, Z_star, rho,
                                                cert_sample, seed)
        dt = time.time() - t0
        row = {
            "c": c, "kappa": round(rho, 4), "resolvent_norm": round(res_norm, 4),
            "sigma1": round(sigma1, 4), "cert_frac": round(cfrac, 4),
            "n_cert": ncert, "n_correct": ncorr, "acc": round(acc, 4),
            "rebuilt": int(rebuilt), "train_s": round(dt, 1),
        }
        out_rows.append(row)
        log(f"  c={c:<5} kappa={rho:.4f} resolvent={res_norm:8.3f} "
            f"sigma1={sigma1:8.3f} cert_frac={cfrac:.3f} ({ncert}/{ncorr}) "
            f"acc={acc:.3f} rebuilt={rebuilt} ({dt:.1f}s)")
        del op
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out_rows


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="seed 42 only, tiny cert sample, 2 c-values; fast sanity")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--cert-sample", type=int, default=800,
                    help="num correct nodes to certify per model (large sample)")
    ap.add_argument("--block-iter", type=int, default=6,
                    help="power-iteration steps for exact per-node block norm "
                         "(converges by ~6 on these operators)")
    ap.add_argument("--block-sample", type=int, default=250,
                    help="num certified nodes to compute EXACT block norm on "
                         "(bounded; proxy covers all). >=250 ample for Spearman")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = print
    log(f"=== exp_coupling_validation === device={device} smoke={args.smoke}")
    if device.type == "cuda":
        log(f"    GPU: {torch.cuda.get_device_name(0)}")

    seeds = [42] if args.smoke else SEEDS
    cert_sample = 200 if args.smoke else args.cert_sample
    epochs = 40 if args.smoke else args.epochs
    block_iter = 5 if args.smoke else args.block_iter
    block_sample = 80 if args.smoke else args.block_sample
    c_values = [0.5, 0.99] if args.smoke else [0.5, 0.7, 0.9, 0.95, 0.99]

    # ---------------- PART (a) ----------------
    if not args.skip_a:
        log("\n=== PART (a): multi-seed coupling correlation (c=0.9 baseline) ===")
        per_seed = []
        for s in seeds:
            per_seed.append(run_part_a_seed(s, device, epochs, cert_sample,
                                            block_iter, log,
                                            block_sample=block_sample))
        # write per-seed per-node CSV (compact: only correlation inputs)
        a_path = RESULTS / ("coupling_validation_partA_smoke.csv" if args.smoke
                            else "coupling_validation_partA.csv")
        with open(a_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "node", "rho_v", "margin_v", "s_exact",
                        "s_proxy", "aL1", "aL2", "aMax"])
            for d in per_seed:
                for k in range(d["n_cert"]):
                    w.writerow([d["seed"], int(d["node_idx"][k]),
                                d["rho_v"][k], d["margin_v"][k],
                                d["s_exact"][k], d["s_proxy"][k],
                                d["aL1"][k], d["aL2"][k], d["aMax"][k]])
        log(f"  wrote {a_path}")

        summary = aggregate_part_a(per_seed, log)
        sum_path = RESULTS / ("coupling_validation_partA_summary_smoke.csv"
                              if args.smoke else
                              "coupling_validation_partA_summary.csv")
        with open(sum_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["correlation", "mean", "std", "min", "max",
                        "p_median", "n_seed"])
            for k, v in summary.items():
                w.writerow([k, f"{v['mean']:+.4f}", f"{v['std']:.4f}",
                            f"{v['min']:+.4f}", f"{v['max']:+.4f}",
                            f"{v['p_med']:.2e}", v["n_seed"]])
        log(f"  wrote {sum_path}")

        log("\n  --- PART (a) summary (mean±std across seeds) ---")
        for k, v in summary.items():
            log(f"    {k:32s} rho = {v['mean']:+.3f} ± {v['std']:.3f}  "
                f"[{v['min']:+.3f},{v['max']:+.3f}]  p_med={v['p_med']:.2e}")

    # ---------------- PART (b) ----------------
    if not args.skip_b:
        log("\n=== PART (b): kappa-divergence sweep (seed 42) ===")
        b_rows = run_part_b(device, c_values, epochs, cert_sample, log)
        b_path = RESULTS / ("coupling_validation_partB_smoke.csv" if args.smoke
                            else "coupling_validation_partB.csv")
        with open(b_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys()))
            w.writeheader()
            for r in b_rows:
                w.writerow(r)
        log(f"  wrote {b_path}")

    log("\n=== done ===")


if __name__ == "__main__":
    main()
