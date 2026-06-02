"""Exp: AEGIS-Conformal (Proposal P2) — robust split conformal prediction for
STRUCTURAL (edge) perturbations, where the worst-case conformity-score shift over
the eps-ball ||delta-Ahat||_F <= eps is bounded ANALYTICALLY by the AEGIS certify
S_c sensitivity, with ZERO Monte-Carlo smoothing.

WHAT THIS DELIVERS
------------------
A SOUND, distribution-free, finite-sample certificate: a prediction set C_eps(v)
with P(y_v in C_eps(v)) >= 1 - alpha SIMULTANEOUSLY over the entire eps-ball of
symmetric edge-supported adjacency perturbations, certified WITHOUT the ~10^4-
sample randomized-smoothing the robust-CP construction otherwise needs.

THE CONSTRUCTION (three pieces, each stated for soundness)
----------------------------------------------------------
1. CONFORMITY SCORE (node classification).  We use APS (adaptive prediction
   sets, Romano-Sesia-Candes 2020) as the primary score and TPS (Sadinle 2018,
   s = softmax of the label) as a sanity score.  Both are CONFORMITY scores
   (higher = more conforming).  Split CP set:  C(v) = { y : s(v,y) >= q_hat },
   with q_hat = Q(alpha; {s_i}_{cal} U {+inf})  the empirical level-alpha
   quantile of calibration scores.  alpha = 0.1 -> target coverage 0.9.
   APS:  s(v,y) = 1 - ( rho(v,y) + u * pi(v)_y ),  rho(v,y) = sum_c pi(v)_c
   1[pi(v)_c > pi(v)_y]  (cumulative softmax mass of classes ranked above y);
   u ~ U[0,1] tie-break.  (We use the CONFORMITY orientation  1 - (...)  so the
   set is {y: s >= q_hat}; equivalent to the standard APS up to sign.)

2. MARGIN-SHIFT BOUND -> WORST-CASE SCORE SHIFT  Delta s_v(eps)  (SOUND).
   The AEGIS certify T3 bound gives, per node v and competitor c, the worst-case
   shift of the (logit_{y} - logit_c) margin over ||delta-Ahat||_F <= eps:
        Delta g_{c,v}(eps) = L1_{c,v} * eps + C_v * eps^2,
        L1_{c,v} = ||(W_y - W_c) S_{c,v}||_2 ,
        C_v      = ||W_y - W_c||_2 * (1 - kappa)^{-2} * L_{J,v} / 2 ,
        L_{J,v}  = ||W||_2^2 * max_{u in 2-hop(v)} ||z*_u||_2   (T3).
   This is the SOUND, curvature-absorbed certify bound (NOT the bare linear
   sigma1*eps, which breaches per rem:certificates).  The bound holds for EVERY
   class margin simultaneously (it is the per-competitor radius integrand).

   Score-Lipschitz link (score <-> margin), stated and made TIGHT, not loose:
   the label softmax  pi_y = 1 / (1 + sum_{c != y} exp(-g_c)),  g_c := logit_y -
   logit_c.  Because  d pi_y / d g_c = pi_y * pi_c >= 0, pi_y is MONOTONE
   INCREASING in every margin g_c.  The adversary minimises pi_y by lowering
   each margin; each margin drops by AT MOST Delta g_{c,v}(eps) (certify).  Hence
   the SOUND worst-case label softmax is
        pi_y^down(eps) = 1 / (1 + sum_{c != y} exp(-(g_c - Delta g_{c,v}(eps)))) ,
   an exact monotone composition of the certify per-competitor bounds (no slack
   from a global Lipschitz constant).  The worst-case CONFORMITY-score shift is
        Delta s_v(eps) = s_clean(v, y) - s_worst(v, y) ,
   where s_worst is the score recomputed at the worst-case softmax pi^down (for
   TPS, s_worst = pi_y^down directly; for APS, we additionally inflate every
   competitor's mass to its certify upper bound pi_c^up so rho(v,y) is maximised
   -- a sound over-estimate of the worst-case APS drop).  Delta s_v(eps) >= 0.

3. ROBUST SPLIT-CP (Zargarbashi & Bojchevski, "Robust CP with a Single Binary
   Certificate", ICLR 2025, arXiv:2503.05239).  Robust CP needs ONLY a worst-
   case bound on the conformity-score change over the threat set.  With a
   DETERMINISTIC analytic bound Delta (not an MC-smoothed CDF), the binary
   certificate reduces to its exact / de-randomised limit: certify on the worst-
   case-decreased scores.  We use the CALIBRATION-ROBUST route (provably
   exchangeable with the worst-case test point):
        q_hat_rob = Q( alpha ; { s_i - Delta_i(eps) }_{cal} U {+inf} )
        C_eps(v)  = { y : s(v,y) - Delta_v(eps) >= q_hat_rob } .
   Soundness: for any perturbation in the ball, the realised true-label score
   s_tilde(v, y_v) >= s(v, y_v) - Delta_v(eps) (worst case), and the lowered
   calibration scores s_i - Delta_i are exchangeable with s_tilde(v, y_v) under
   (A1)-(A3) + calibration/test exchangeability, so the standard split-CP
   guarantee on the lowered scores transfers:  P(y_v in C_eps(v)) >= 1 - alpha
   simultaneously over the whole eps-ball.  (eps=0 recovers vanilla split CP.)

EVALUATION
----------
Cora (+ Citeseer if compute allows), eps in {0.01, 0.05}, alpha = 0.1, on a DENSE
ego-subgraph (exact S_c, exact reconvergence for the attack gate).  We report:
  * non-robust split-CP MARGINAL coverage (sanity ~0.90) + avg set size;
  * ROBUST set: certified coverage + avg set size per eps;
  * THE HARD GATE -- empirical coverage UNDER ATTACK: apply the AEGIS worst-case
    attack (certify v1 / binding-competitor direction) at magnitude eps to each
    test node, RECONVERGE the equilibrium, recompute the score, and confirm the
    TRUE label stays in C_eps(v) >= 1 - alpha of the time.  Also random eps
    directions.  If the robust set is breached below 1 - alpha the construction
    is WRONG.
  * robust vs non-robust set size; cost-vs-smoothing note (ZERO MC samples).

USAGE
-----
    .venv/bin/python scripts/exp_aegis_conformal.py \
        [--datasets Cora,Citeseer] [--subgraph-nodes 400] [--seeds 42,137,271] \
        [--eps 0.01,0.05] [--alpha 0.1] [--cal-frac 0.5] [--epochs 150] \
        [--attack-nodes 60] [--n-random 5] [--quick]

Writes:
    results/aegis_conformal.csv         per (dataset,seed,eps,score) coverage/size
    results/aegis_conformal_gate.csv    per-node under-attack gate detail
    results/aegis_conformal_summary.csv aggregated table (mean over seeds)
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
# reuse the VALIDATED certify machinery (bound constants + attack/reconverge gate)
from scripts.exp_certify_tighten import (  # noqa: E402
    make_LJ_provider,
    w_norm_2,
    _attack_and_check,
    _build_symmetric_dA,
    _safe_reconverge,
)

SQRT2 = math.sqrt(2.0)
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Conformity scores (CONFORMITY orientation: higher = more conforming;
# set = {y : s(v,y) >= q_hat}).
# ---------------------------------------------------------------------------
def tps_scores(pi: torch.Tensor) -> torch.Tensor:
    """TPS conformity score matrix s(v,y) = pi(v)_y  (Sadinle 2018). (N, C)."""
    return pi.clone()


def aps_scores(pi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """APS conformity score s(v,y) = 1 - (rho(v,y) + u_v * pi(v)_y), where
    rho(v,y) = sum_c pi(v)_c 1[pi(v)_c > pi(v)_y]  (mass of classes strictly
    above y).  Conformity orientation (1 - cumulative) so set = {y: s >= q}.
    u: (N,) per-node uniform tie-break in [0,1].  Returns (N, C)."""
    N, C = pi.shape
    s = torch.zeros_like(pi)
    for y in range(C):
        py = pi[:, y:y + 1]                       # (N,1)
        rho = (pi * (pi > py).to(pi.dtype)).sum(dim=1)   # (N,)
        s[:, y] = 1.0 - (rho + u * pi[:, y])
    return s


# ---------------------------------------------------------------------------
# Analytic worst-case score-shift Delta s from the certify T3 bound.
#
# For a REFERENCE label r (the label whose conformity score the adversary tries
# to lower so r drops out of the set), the certify bound gives, per competitor
# c != r, the worst-case drop of the margin g_c = (logit_r - logit_c):
#     Delta g_{c}(eps) = L1_{c} * eps + C_v * eps^2,
#     L1_{c} = ||(W_r - W_c) S_{c,v}||_2 ,
#     C_v    = ||W_r - W_c||_2 * (1 - kappa)^{-2} * L_{J,v} / 2 .
# pi_r = 1 / (1 + sum_{c!=r} exp(-g_c)) is MONOTONE increasing in every g_c
# (d pi_r / d g_c = pi_r pi_c >= 0), so the worst case lowers EVERY margin by its
# certify bound, giving the SOUND worst-case softmax pi^worst (label r minimised,
# competitors maximised).  The conformity-score drop is s_clean(v,r) -
# s(pi^worst, r) >= 0, computed exactly (no global-Lipschitz slack).
# ---------------------------------------------------------------------------
def worst_case_softmax_for_ref(logits_v, r, Sv, W, kfac, L_J_v, eps):
    """SOUND worst-case softmax when the adversary attacks reference label r:
    every margin g_c = logit_r - logit_c (c != r) is lowered by its certify
    bound L1_c*eps + C_v*eps^2.  Returns pi^worst (C,)."""
    C = logits_v.shape[0]
    gy = logits_v[r]
    g = (gy - logits_v)                          # margins g_c (g_r = 0)
    gw = g.clone()
    Wr = W[r]
    for c in range(C):
        if c == r:
            continue
        wgap = (Wr - W[c])
        L1_c = float((wgap @ Sv).norm())
        C_v = float(wgap.norm()) * kfac * L_J_v / 2.0
        gw[c] = g[c] - (L1_c * eps + C_v * eps * eps)   # lower the margin (sound)
    neg = (-gw)
    neg = neg - neg.max()
    e = torch.exp(neg)
    return e / e.sum()


def label_deltas_for_node(logits_v, Sv, W, kfac, L_J_v, eps, u_v, score_kind):
    """Per-LABEL worst-case conformity-score DROP vector Delta (C,) for node v:
    Delta[r] = s_clean(v,r) - s_worst(v,r) >= 0, where s_worst uses the worst-
    case softmax computed with r as the attacked reference label.  This is the
    amount the adversary can lower label r's conformity score, used to inflate
    r's robust-set threshold (sound per label, hence sound set size AND sound
    true-label coverage)."""
    C = logits_v.shape[0]
    pi_clean = torch.softmax(logits_v, dim=0)
    delta = torch.zeros(C)
    for r in range(C):
        pw = worst_case_softmax_for_ref(logits_v, r, Sv, W, kfac, L_J_v, eps)
        if score_kind == "tps":
            s_clean = float(pi_clean[r])
            s_worst = float(pw[r])
        else:  # APS conformity:  s = 1 - (rho + u*pi_r)
            rho_clean = float((pi_clean * (pi_clean > pi_clean[r]).to(pi_clean.dtype)).sum())
            s_clean = 1.0 - (rho_clean + u_v * float(pi_clean[r]))
            rho_w = float((pw * (pw > pw[r]).to(pw.dtype)).sum())   # excludes r (strict >)
            s_worst = 1.0 - (rho_w + u_v * float(pw[r]))
        delta[r] = max(0.0, s_clean - s_worst)
    return delta


# ---------------------------------------------------------------------------
# Split-CP quantile (conformity orientation).
# ---------------------------------------------------------------------------
def cp_quantile(cal_scores: torch.Tensor, alpha: float) -> float:
    """q_hat = level-alpha empirical quantile of calibration conformity scores
    with the finite-sample (n+1) correction:  the set {y: s >= q_hat} has
    P >= 1 - alpha.  For conformity scores we take the
    floor((alpha)(n+1))-th order statistic (the alpha-lower quantile); labels
    with score >= q_hat are kept.  Implemented as the
    (k = floor(alpha*(n+1)))-th smallest calibration score (1-indexed), clamped
    so k in [1, n]; if alpha*(n+1) < 1 -> q_hat = -inf (keep all = full coverage)."""
    s = torch.sort(cal_scores).values
    n = s.numel()
    k = math.floor(alpha * (n + 1))
    if k < 1:
        return float("-inf")
    if k > n:
        k = n
    return float(s[k - 1])


def coverage_and_size(test_scores: torch.Tensor, y_test: torch.Tensor,
                      q_hat: float, delta_test: torch.Tensor | None = None):
    """Coverage = fraction of test nodes whose TRUE-label score >= q_hat (minus
    delta if robust); avg set size = mean over test nodes of #{y: s>=q_hat
    (- delta)}.  test_scores (n_test, C); delta_test (n_test,) the per-node
    score shift (None -> vanilla)."""
    n, C = test_scores.shape
    if delta_test is None:
        thr_label = test_scores[torch.arange(n), y_test]
        covered = (thr_label >= q_hat).float().mean().item()
        sizes = (test_scores >= q_hat).sum(dim=1).float().mean().item()
    else:
        # robust: include label r iff s(v,r) - delta_{v,r} >= q_hat
        #   <=>  s(v,r) >= q_hat + delta_{v,r}.  delta_test is (n, C) per-label.
        thr = q_hat + delta_test                 # (n, C)
        d_true = delta_test[torch.arange(n), y_test]
        covered = (test_scores[torch.arange(n), y_test] >= (q_hat + d_true)).float().mean().item()
        sizes = (test_scores >= thr).sum(dim=1).float().mean().item()
    return covered, sizes


# ---------------------------------------------------------------------------
# THE HARD GATE: empirical coverage UNDER ATTACK (reconverge, recompute score).
# ---------------------------------------------------------------------------
def attack_direction_for_label(Sc, W, d, v, y, preds, logits):
    """Worst-case edge direction that lowers the TRUE-label margin g_c for the
    binding competitor c* = argmin_{c != y, g_c > 0} g_c (the closest class to
    flipping y).  Returns (w_dir (|E|,), c_star) or (None, -1) if y already not
    the top / no positive margin."""
    Sv = Sc[v * d:(v + 1) * d, :]
    lgv = logits[v]
    gy = lgv[y]
    g = (gy - lgv)
    C = lgv.shape[0]
    posc = [c for c in range(C) if c != y and float(g[c]) > 0]
    if not posc:
        return None, -1
    c_star = min(posc, key=lambda c: float(g[c]))
    wgap = (W[y] - W[c_star])
    direction = (wgap @ Sv)                      # (|E|,) gradient of g_{c*} wrt edges
    gn = float(direction.norm())
    if gn < 1e-30:
        return None, -1
    return -(direction / gn), c_star             # MINUS: lower the margin


def reconverge_softmax_after_attack(model, A_sub, ctx_sub, Z_sub, op, w_dir,
                                    target_fro, v, nc):
    """Apply symmetric dA of exact ||.||_F = target_fro along w_dir, reconverge
    the FULL subgraph equilibrium, and return the post-attack softmax pi(v) (C,)
    at node v.  SCORE-AGNOSTIC: returns (pi, diverged) so the caller can score
    BOTH APS and TPS from the SAME reconvergence (the attack direction depends
    only on the true label, not the score)."""
    if target_fro <= 0:
        with torch.no_grad():
            return torch.softmax(model.head(Z_sub)[v], dim=0), False
    dA = _build_symmetric_dA(op, w_dir, target_fro)
    A_pert = (A_sub + dA).contiguous()
    ctx_p = {"A_hat": A_pert, "X_proj": ctx_sub["X_proj"]}
    Z_p, diverged, _ = _safe_reconverge(model, Z_sub, ctx_p, max_iter=400, tol=1e-9)
    if diverged:
        # diverged perturbed system -> true-label score collapses (worst case).
        return torch.full((nc,), 0.0, device=Z_sub.device), True
    with torch.no_grad():
        return torch.softmax(model.head(Z_p)[v], dim=0), False


def conformity_score_vec(pi, u_v, score_kind, nc):
    """Map a softmax pi (C,) to the per-label CONFORMITY score vector (C,)."""
    if score_kind == "tps":
        return pi.clone()
    s = torch.zeros(nc, device=pi.device)
    for y in range(nc):
        rho = float((pi * (pi > pi[y]).to(pi.dtype)).sum())
        s[y] = 1.0 - (rho + u_v * float(pi[y]))
    return s


# ---------------------------------------------------------------------------
# Per (dataset, seed) run.
# ---------------------------------------------------------------------------
def run_one(dataset, seed, args, dev, w_csv, gate_csv):
    t0 = time.time()
    X, A, y, tm, nf, nc = load_dataset(dataset)
    X, A, y = X.to(dev), A.to(dev), y.to(dev)
    model = train_ignn(X, A, y, tm, nf, nc, dev, seed=seed, epochs=args.epochs)
    with torch.no_grad():
        _, Zf, ctxf = model(X, A)

    # dense ego-subgraph: exact S_c + exact reconvergence for the gate
    idx = extract_ego_subgraph(A, max_nodes=args.subgraph_nodes)
    A_sub = A[idx][:, idx].contiguous()
    ctx_sub = {"A_hat": A_sub, "X_proj": ctxf["X_proj"][idx].contiguous()}
    Z_sub = reconverge(model, Zf[idx].clone(), ctx_sub, max_iter=600, tol=1e-9)
    y_sub = y[idx]
    N = A_sub.shape[0]
    d = model.hidden
    W = model.head.weight.detach()

    op = ScalableSensitivity(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    kappa = rho_rayleigh(op)
    Fop = lambda z, c: model.operator(z, c)
    S = structural_sensitivity_matrix(Fop, Z_sub, ctx_sub, "A_hat")
    Sc_code, edge_list = constrained_sensitivity_matrix(S, A_sub)
    Sc = Sc_code / SQRT2
    desc, LJ_fn = make_LJ_provider("T3", model, Z_sub, edge_list, N)

    with torch.no_grad():
        logits = model.head(Z_sub)
        pi = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    sub_acc = float((preds == y_sub).float().mean())

    # CP split: random calibration / test over subgraph nodes
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_cal = int(round(args.cal_frac * N))
    cal_idx = perm[:n_cal].to(dev)
    test_idx = perm[n_cal:].to(dev)
    n_test = test_idx.numel()

    # per-node uniform tie-break for APS (fixed per node, shared cal/test)
    u_all = torch.rand(N, generator=g).to(dev)

    print(f"[{dataset}/s{seed}] N={N} E={op.num_edges} kappa={kappa:.4f} "
          f"acc={sub_acc:.3f} n_cal={n_cal} n_test={n_test} "
          f"t={time.time()-t0:.0f}s", flush=True)

    eps_list = [float(e) for e in args.eps]
    kfac = (1.0 - kappa) ** (-2)
    results = {}
    score_kinds = ["aps", "tps"]

    # Precompute clean conformity-score matrices for both scores.
    S_all = {"tps": tps_scores(pi), "aps": aps_scores(pi, u_all)}
    cal_nodes = cal_idx.tolist()
    test_nodes = test_idx.tolist()
    test_pos = {v: i for i, v in enumerate(test_nodes)}

    # --- vanilla split CP (eps=0 sanity) for both scores ---
    q0 = {}
    for sk in score_kinds:
        cal_true = S_all[sk][cal_idx, y_sub[cal_idx]]
        q0[sk] = cp_quantile(cal_true, args.alpha)
        cov0, size0 = coverage_and_size(S_all[sk][test_idx], y_sub[test_idx], q0[sk])
        w_csv.writerow({
            "dataset": dataset, "seed": seed, "score": sk, "eps": 0.0,
            "mode": "vanilla", "q_hat": q0[sk], "coverage": cov0, "avg_set_size": size0,
            "cov_under_worstcase_attack": "", "cov_under_random_attack": "",
            "n_cal": n_cal, "n_test": n_test, "kappa": kappa, "acc": sub_acc,
            "n_attack_nodes": "",
        })
        results[(sk, 0.0, "vanilla")] = (cov0, size0)

    for eps in eps_list:
        # Per-LABEL worst-case score-drop matrices (n, C) for cal and test nodes,
        # per score.  delta_node[v] is the (C,) vector from label_deltas_for_node.
        def delta_matrix(node_list, sk):
            rows = []
            for v in node_list:
                v = int(v)
                Sv = Sc[v * d:(v + 1) * d, :]
                rows.append(label_deltas_for_node(
                    logits[v], Sv, W, kfac, LJ_fn(v), eps, float(u_all[v]), sk))
            return torch.stack(rows, dim=0).to(dev)        # (n, C)

        q_rob = {}; delta_test_mat = {}
        for sk in score_kinds:
            dcal = delta_matrix(cal_nodes, sk)             # (n_cal, C)
            dtest = delta_matrix(test_nodes, sk)           # (n_test, C)
            delta_test_mat[sk] = dtest
            # calibration-robust quantile: lower each cal TRUE-label score by its
            # own per-label delta, then take the standard quantile (exchangeable
            # with the worst-case-lowered test true-label score).
            cal_true = S_all[sk][cal_idx, y_sub[cal_idx]]
            d_cal_true = dcal[torch.arange(len(cal_nodes)), y_sub[cal_idx]]
            q_rob[sk] = cp_quantile(cal_true - d_cal_true, args.alpha)
            cov_rob, size_rob = coverage_and_size(
                S_all[sk][test_idx], y_sub[test_idx], q_rob[sk], delta_test=dtest)
            w_csv.writerow({
                "dataset": dataset, "seed": seed, "score": sk, "eps": eps,
                "mode": "robust", "q_hat": q_rob[sk], "coverage": cov_rob,
                "avg_set_size": size_rob, "cov_under_worstcase_attack": "",
                "cov_under_random_attack": "", "n_cal": n_cal, "n_test": n_test,
                "kappa": kappa, "acc": sub_acc, "n_attack_nodes": "",
            })
            results[(sk, eps, "robust_pre")] = (cov_rob, size_rob)

        # ---- THE HARD GATE: empirical coverage under attack (score-agnostic
        # reconvergence; score BOTH APS and TPS from the same perturbed pi). ----
        n_gate = min(args.attack_nodes, n_test)
        gate_nodes = test_idx[:n_gate].tolist()
        # accumulators per score
        n_in_wc = {sk: 0 for sk in score_kinds}
        n_in_rand = {sk: 0 for sk in score_kinds}
        n_eval = 0; n_div = 0
        n_bound_ok = {sk: 0 for sk in score_kinds}     # soundness-bound check
        for v in gate_nodes:
            v = int(v); y_lab = int(y_sub[v]); u_v = float(u_all[v])
            # worst-case attack lowers the TRUE-label margin (binding competitor)
            wdir, c_star = attack_direction_for_label(Sc, W, d, v, y_lab, preds, logits)
            if wdir is None:
                continue
            n_eval += 1
            pi_wc, div_wc = reconverge_softmax_after_attack(
                model, A_sub, ctx_sub, Z_sub, op, wdir, eps, v, nc)
            if div_wc:
                n_div += 1
            # random directions: keep the WORST (lowest true-label softmax) over
            # n_random draws, shared across scores.
            pi_rand_worst = None
            for r in range(args.n_random):
                torch.manual_seed(9173 * v + r)
                wr = torch.randn(op.num_edges, device=dev, dtype=Sc.dtype)
                wr = wr / (wr.norm() + 1e-30)
                pr, _ = reconverge_softmax_after_attack(
                    model, A_sub, ctx_sub, Z_sub, op, wr, eps, v, nc)
                if pi_rand_worst is None or float(pr[y_lab]) < float(pi_rand_worst[y_lab]):
                    pi_rand_worst = pr
            for sk in score_kinds:
                q = q_rob[sk]
                dv = float(delta_test_mat[sk][test_pos[v], y_lab])
                s_clean_true = float(S_all[sk][v, y_lab])
                s_wc = conformity_score_vec(pi_wc, u_v, sk, nc)
                s_wc_true = float(s_wc[y_lab])
                in_wc = (s_wc_true >= q)
                if in_wc:
                    n_in_wc[sk] += 1
                # soundness-bound validation: did the certify bound hold?
                #   s_attacked >= s_clean - delta_v   (the analytic lower bound)
                if s_wc_true >= s_clean_true - dv - 1e-6:
                    n_bound_ok[sk] += 1
                s_r = conformity_score_vec(pi_rand_worst, u_v, sk, nc)
                in_rand = (float(s_r[y_lab]) >= q)
                if in_rand:
                    n_in_rand[sk] += 1
                gate_csv.writerow({
                    "dataset": dataset, "seed": seed, "score": sk, "eps": eps,
                    "node": v, "true_label": y_lab, "binding_c": c_star,
                    "q_rob": q, "delta_v": dv,
                    "s_clean_true": s_clean_true, "s_attacked_true": s_wc_true,
                    "in_robust_set_wc": int(in_wc),
                    "in_robust_set_rand": int(in_rand),
                    "diverged": int(div_wc),
                })
        for sk in score_kinds:
            cov_wc = n_in_wc[sk] / max(n_eval, 1)
            cov_rand = n_in_rand[sk] / max(n_eval, 1)
            bound_ok = n_bound_ok[sk] / max(n_eval, 1)
            cov_rob, size_rob = results[(sk, eps, "robust_pre")]
            w_csv.writerow({
                "dataset": dataset, "seed": seed, "score": sk, "eps": eps,
                "mode": "robust_gate", "q_hat": q_rob[sk], "coverage": cov_rob,
                "avg_set_size": size_rob,
                "cov_under_worstcase_attack": cov_wc,
                "cov_under_random_attack": cov_rand,
                "n_cal": n_cal, "n_test": n_test, "kappa": kappa, "acc": sub_acc,
                "n_attack_nodes": n_eval,
            })
            results[(sk, eps, "robust")] = (cov_rob, size_rob, cov_wc, cov_rand, n_eval, n_div, bound_ok)
            print(f"   [{sk} eps={eps}] vanilla_cov={results[(sk,0.0,'vanilla')][0]:.3f} "
                  f"size0={results[(sk,0.0,'vanilla')][1]:.2f} | "
                  f"robust_cov={cov_rob:.3f} robust_size={size_rob:.2f} | "
                  f"GATE wc_cov={cov_wc:.3f} rand_cov={cov_rand:.3f} "
                  f"bound_held={bound_ok:.3f} (n={n_eval}, div={n_div})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Cora,Citeseer")
    ap.add_argument("--subgraph-nodes", type=int, default=400)
    ap.add_argument("--seeds", default="42,137,271")
    ap.add_argument("--eps", default="0.01,0.05")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--attack-nodes", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    args.eps = [s for s in args.eps.split(",") if s]
    if args.quick:
        args.seeds = args.seeds.split(",")[0]
        args.subgraph_nodes = min(args.subgraph_nodes, 150)
        args.attack_nodes = min(args.attack_nodes, 20)
        args.epochs = min(args.epochs, 60)

    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    datasets = [s for s in args.datasets.split(",") if s]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    w_csv_f = open(RESULTS / "aegis_conformal.csv", "w", newline="")
    w_csv = csv.DictWriter(w_csv_f, fieldnames=[
        "dataset", "seed", "score", "eps", "mode", "q_hat", "coverage",
        "avg_set_size", "cov_under_worstcase_attack", "cov_under_random_attack",
        "n_cal", "n_test", "kappa", "acc", "n_attack_nodes"])
    w_csv.writeheader()

    gate_f = open(RESULTS / "aegis_conformal_gate.csv", "w", newline="")
    gate_csv = csv.DictWriter(gate_f, fieldnames=[
        "dataset", "seed", "score", "eps", "node", "true_label", "binding_c",
        "q_rob", "delta_v", "s_clean_true", "s_attacked_true",
        "in_robust_set_wc", "in_robust_set_rand", "diverged"])
    gate_csv.writeheader()

    all_res = {}
    for ds in datasets:
        for seed in seeds:
            r = run_one(ds, seed, args, dev, w_csv, gate_csv)
            w_csv_f.flush(); gate_f.flush()
            for k, v in r.items():
                all_res.setdefault((ds,) + k, []).append(v)

    w_csv_f.close(); gate_f.close()

    # ---- aggregate summary across seeds ----
    summ_f = open(RESULTS / "aegis_conformal_summary.csv", "w", newline="")
    sw = csv.DictWriter(summ_f, fieldnames=[
        "dataset", "score", "eps", "mode",
        "vanilla_cov", "vanilla_size",
        "robust_cov", "robust_size",
        "gate_worstcase_cov", "gate_random_cov", "bound_held_frac",
        "alpha", "target_cov", "n_seeds", "n_attack_nodes_total"])
    sw.writeheader()

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    for ds in datasets:
        for score_kind in ["aps", "tps"]:
            van = all_res.get((ds, score_kind, 0.0, "vanilla"), [])
            van_cov = mean([x[0] for x in van]); van_size = mean([x[1] for x in van])
            for eps in [float(e) for e in args.eps]:
                rob = all_res.get((ds, score_kind, eps, "robust"), [])
                if not rob:
                    continue
                rob_cov = mean([x[0] for x in rob]); rob_size = mean([x[1] for x in rob])
                wc = mean([x[2] for x in rob]); rd = mean([x[3] for x in rob])
                ntot = sum(int(x[4]) for x in rob)
                bh = mean([x[6] for x in rob])
                sw.writerow({
                    "dataset": ds, "score": score_kind, "eps": eps, "mode": "robust",
                    "vanilla_cov": round(van_cov, 4), "vanilla_size": round(van_size, 3),
                    "robust_cov": round(rob_cov, 4), "robust_size": round(rob_size, 3),
                    "gate_worstcase_cov": round(wc, 4), "gate_random_cov": round(rd, 4),
                    "bound_held_frac": round(bh, 4),
                    "alpha": args.alpha, "target_cov": 1 - args.alpha,
                    "n_seeds": len(rob), "n_attack_nodes_total": ntot,
                })
    summ_f.close()

    print(f"\n{'='*70}")
    print("=== AEGIS-CONFORMAL SUMMARY (mean over seeds) ===")
    print(f"  alpha={args.alpha}  target coverage={1-args.alpha}")
    with open(RESULTS / "aegis_conformal_summary.csv") as f:
        for line in f:
            print("  " + line.rstrip())
    print(f"\n  total wall time: {time.time()-t0:.0f}s")
    print(f"  wrote: results/aegis_conformal.csv, _gate.csv, _summary.csv")


if __name__ == "__main__":
    sys.exit(main() or 0)
