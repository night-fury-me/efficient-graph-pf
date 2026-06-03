"""Exp: AEGIS-Conformal (analytic, ZERO-sample) vs randomized-smoothing certifier,
HEAD-TO-HEAD on the SAME nodes — turns the paper's "no Monte-Carlo smoothing" claim
from an assertion into a MEASURED comparison (coverage, set size, wall-clock).

WHY
---
AEGIS-Conformal (scripts/exp_aegis_conformal.py) builds a robust split-CP prediction
set C_eps(v) with P(y_v in C_eps(v)) >= 1-alpha SIMULTANEOUSLY over the whole eps-ball
of symmetric edge perturbations, using the ANALYTIC S_c certify bound and ZERO Monte-
Carlo samples. The competing route is Cohen-style randomized smoothing: draw M noisy
adjacencies, reconverge+classify each, and certify a radius from a Clopper-Pearson lower
bound on the top-class probability. Smoothing pays M forward+reconverge passes per node.
This script runs BOTH on the same n=200 Cora subgraph and the same test nodes, so the
cost/coverage/size trade-off is measured, not asserted. HONESTY: if smoothing wins a
metric (e.g. smaller sets), we report it.

THE TWO ARMS (same model, same calibration/test split, same eps, same alpha)
----------------------------------------------------------------------------
* AEGIS-CONFORMAL (reused verbatim from exp_aegis_conformal.py):
    per-label worst-case conformity-score drop Delta_v(eps) from the certify T3 bound,
    calibration-robust quantile q_rob, robust set C_eps(v) = {y : s(v,y)-Delta_v >= q_rob}.
    Wall-clock = the per-label Delta build + CP-set construction (analytic; NO sampling,
    NO reconvergence). The one-time S_c factorisation is shared infrastructure (both arms
    need the subgraph + model); we time only the certifier-specific construction.

* RANDOMIZED SMOOTHING (new, Cohen-Rosenfeld-Kolter 2019 adapted to the structural ball):
    sigma<->eps MATCHING. We smooth over symmetric edge-supported Gaussian noise: for each
    sample, xi ~ N(0, sigma^2 I_{|E|}) on the |E| upper-triangular edge coordinates, mirrored
    to the lower triangle (op._edges_to_delta_A), giving a symmetric dA. Because each edge is
    written twice, E[||dA||_F^2] = 2|E| sigma^2; matching this to the eps-ball radius
    (E||dA||_F^2 = eps^2) gives
                      sigma = eps / sqrt(2|E|).
    (Verified empirically: mean ||dA||_F ~ eps.) Per node v: draw M samples, reconverge the
    IGNN equilibrium under each (warm-started from Z*, contractive operator -> ~10 iters),
    classify (argmax of the readout at v). Let c_A = empirically-top class, n_A its count.
    Clopper-Pearson lower bound pA_lo = Beta^{-1}(alpha; n_A, M-n_A+1) (one-sided, conf 1-alpha).
    Cohen certified radius  R = sigma * Phi^{-1}(pA_lo).  PREDICTION SET with a 1-alpha-style
    coverage guarantee over the eps-ball:
        - if pA_lo > 1/2 AND R >= eps:  c_A is CERTIFIED to remain the smoothed top class
          over the whole eps-ball -> set = {c_A}  (size 1; covers y_v iff c_A == y_v).
        - else: the smoothed top class is NOT certified at radius eps -> ABSTAIN to the
          conservative full label set {0..C-1} (size C) so coverage is retained (a vacuous
          but SOUND fallback; this is the honest price of an uncertified node).
      This is the standard Cohen guarantee turned into a coverage set: the certified set is
      the singleton top class when certifiable at eps, else everything. Wall-clock = M x
      (reconverge + classify) per node; we run a feasible smoke M and EXTRAPOLATE to M=1e4.

SELF-CHECKS (asserted at runtime; failures print FAIL but do not crash)
-----------------------------------------------------------------------
 1. both arms hit coverage >= 1-alpha at eps=0.01 (else mis-calibrated).
 2. smoothing wall-clock >> conformal wall-clock (the expected win).
 3. the smoothed top-class probability is a valid probability in [0,1], and pA_lo <= pA_hat.

OUTPUT
------
A compact table {method x eps -> coverage, mean set size, wall-clock (smoke + extrap 1e4)},
printed and written to results/conformal_vs_smoothing.csv. Seed-42 Cora smoke ONLY.

USAGE
-----
    ./.venv/bin/python scripts/exp_conformal_vs_smoothing.py \
        [--subgraph-nodes 200] [--seed 42] [--eps 0.01,0.05] [--alpha 0.1] \
        [--cal-frac 0.5] [--epochs 150] [--smoothing-nodes 60] [--M 200] \
        [--smooth-tol 1e-5] [--smooth-maxiter 60] [--extrap-M 10000]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
from scipy.stats import beta as beta_dist, norm as normal_dist

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
from scripts.exp_certify_tighten import make_LJ_provider  # noqa: E402
# REUSE the bug-audited AEGIS-Conformal machinery verbatim.
from scripts.exp_aegis_conformal import (  # noqa: E402
    tps_scores,
    aps_scores,
    label_deltas_for_node,
    cp_quantile,
    coverage_and_size,
)

SQRT2 = math.sqrt(2.0)
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Randomized-smoothing certifier (Cohen et al. 2019), structural-ball variant.
# ---------------------------------------------------------------------------
def cp_lower_bound(n_A: int, M: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound (conf 1-alpha) on the
    Binomial success probability given n_A successes out of M.  Cohen et al. use
    exactly this for the top-class probability.  Returns pA_lo in [0,1]."""
    if n_A <= 0:
        return 0.0
    if n_A >= M:
        # exact one-sided lower bound when all samples agree
        return float(alpha ** (1.0 / M))
    return float(beta_dist.ppf(alpha, n_A, M - n_A + 1))


def smoothed_certificate(counts: torch.Tensor, M: int, sigma: float,
                         eps: float, alpha: float):
    """Given per-class sample counts (C,) from M smoothing draws, return the
    Cohen certificate and the coverage prediction set.

    Returns dict with:
      c_A        : empirically-top class
      pA_hat     : n_A / M   (valid probability in [0,1])
      pA_lo      : CP lower bound on P(f=c_A)
      radius     : sigma * Phi^{-1}(pA_lo)  (>=0 iff pA_lo>=1/2; -inf if pA_lo=0)
      certified  : bool, top class certified to survive the WHOLE eps-ball
      pred_set   : set of class indices (singleton {c_A} if certified, else all)
    """
    C = counts.numel()
    n_A = int(counts.max().item())
    c_A = int(counts.argmax().item())
    pA_hat = n_A / M
    pA_lo = cp_lower_bound(n_A, M, alpha)
    if pA_lo <= 0.5:
        radius = float("-inf") if pA_lo <= 0.0 else 0.0
        certified = False
    else:
        radius = sigma * float(normal_dist.ppf(pA_lo))
        certified = radius >= eps
    pred_set = {c_A} if certified else set(range(C))
    return {
        "c_A": c_A, "pA_hat": pA_hat, "pA_lo": pA_lo,
        "radius": radius, "certified": certified, "pred_set": pred_set,
    }


def smoothing_classify_node(model, A_sub, ctx_sub, Z_sub, edge_idx, v, nc,
                            M, sigma, dev, tol, max_iter):
    """Draw M symmetric edge-noise perturbations, reconverge the equilibrium under
    each (warm-started from Z_sub), classify node v.  Returns class-count vector
    (nc,).  This is the per-node smoothing forward pass (the costly part)."""
    E = edge_idx.shape[0]
    counts = torch.zeros(nc, device=dev)
    Xp = ctx_sub["X_proj"]
    with torch.no_grad():
        for _ in range(M):
            xi = torch.randn(E, device=dev, dtype=A_sub.dtype) * sigma
            dA = torch.zeros_like(A_sub)
            dA[edge_idx[:, 0], edge_idx[:, 1]] = xi
            dA[edge_idx[:, 1], edge_idx[:, 0]] = xi
            A_pert = (A_sub + dA).contiguous()
            ctx_p = {"A_hat": A_pert, "X_proj": Xp}
            Z = Z_sub.clone()
            for _it in range(max_iter):
                Zn = model.operator(Z, ctx_p)
                if (Zn - Z).norm() < tol * max(float(Z.norm()), 1.0):
                    Z = Zn
                    break
                Z = Zn
            cls = int(model.head(Z)[v].argmax().item())
            counts[cls] += 1
    return counts


# ---------------------------------------------------------------------------
# Main: one seed-42 Cora subgraph, both arms on the same test nodes.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--subgraph-nodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", default="0.01,0.05")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--smoothing-nodes", type=int, default=60)
    ap.add_argument("--M", type=int, default=200,
                    help="smoke smoothing samples per node (extrapolated to --extrap-M)")
    ap.add_argument("--smooth-tol", type=float, default=1e-5)
    ap.add_argument("--smooth-maxiter", type=int, default=60)
    ap.add_argument("--extrap-M", type=int, default=10000)
    ap.add_argument("--score", default="aps", choices=["aps", "tps"],
                    help="conformity score for the AEGIS-Conformal arm")
    ap.add_argument("--sigma-match", default="frob,per_edge",
                    help="comma list of sigma<->eps matchings for the smoothing arm. "
                         "'frob' = sigma=eps/sqrt(2|E|) (the SAME Frobenius eps-ball as "
                         "AEGIS, apples-to-apples). 'per_edge' = sigma=eps (treat eps as a "
                         "per-coordinate l2 std; a LARGER threat radius, favorable to "
                         "smoothing). Both reported for honesty.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = cuda if a free GPU exists else cpu. The exact dense S_c "
                         "build needs ~2GB; on a contended GPU pass cpu. BOTH arms run on "
                         "the SAME device so the wall-clock comparison stays fair.")
    ap.add_argument("--out", default="conformal_vs_smoothing.csv",
                    help="output CSV filename under results/ (unique name lets parallel "
                         "per-seed workers avoid clobbering each other).")
    args = ap.parse_args()
    eps_list = [float(e) for e in args.eps.split(",") if e]
    alpha = args.alpha
    sk = args.score

    if args.device == "cpu":
        dev = torch.device("cpu")
    elif args.device == "cuda":
        dev = torch.device("cuda")
    else:  # auto: use cuda only if >=3GB free (dense S_c build needs ~2GB), else cpu
        dev = torch.device("cpu")
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            if free_b > 3 * (1024 ** 3):
                dev = torch.device("cuda")
    print(f"[device] {dev}", flush=True)
    t_all = time.time()

    # ----- train IGNN + build n=200 subgraph (EXACTLY as exp_aegis_conformal) -----
    X, A, y, tm, nf, nc = load_dataset(args.dataset)
    X, A, y = X.to(dev), A.to(dev), y.to(dev)
    model = train_ignn(X, A, y, tm, nf, nc, dev, seed=args.seed, epochs=args.epochs)
    with torch.no_grad():
        _, Zf, ctxf = model(X, A)
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
    _desc, LJ_fn = make_LJ_provider("T3", model, Z_sub, edge_list, N)
    edge_idx = op._edge_idx  # (E,2) upper-triangular active edges
    E = op.num_edges

    with torch.no_grad():
        logits = model.head(Z_sub)
        pi = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    sub_acc = float((preds == y_sub).float().mean())

    # ----- calibration / test split (EXACTLY as exp_aegis_conformal) -----
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_cal = int(round(args.cal_frac * N))
    cal_idx = perm[:n_cal].to(dev)
    test_idx = perm[n_cal:].to(dev)
    n_test = test_idx.numel()
    u_all = torch.rand(N, generator=g).to(dev)
    kfac = (1.0 - kappa) ** (-2)

    print(f"[{args.dataset}/s{args.seed}] N={N} E={E} kappa={kappa:.4f} acc={sub_acc:.3f} "
          f"nc={nc} n_cal={n_cal} n_test={n_test} score={sk} "
          f"sigma<->eps: sigma=eps/sqrt(2|E|), |E|={E}", flush=True)

    # clean conformity-score matrix for the chosen score
    S_clean = {"tps": tps_scores(pi), "aps": aps_scores(pi, u_all)}[sk]
    cal_nodes = cal_idx.tolist()
    test_nodes = test_idx.tolist()

    # smoothing evaluated on the FIRST k test nodes (same nodes both arms compare on)
    n_sm = min(args.smoothing_nodes, n_test)
    sm_nodes = test_nodes[:n_sm]
    sm_node_set = test_idx[:n_sm]
    y_sm = y_sub[sm_node_set]

    matchings = [m for m in args.sigma_match.split(",") if m]
    rows = []          # output table rows
    selfcheck = []     # (name, ok, detail)

    for eps in eps_list:
        # ============================ AEGIS-CONFORMAL ============================
        # Analytic, ZERO-sample. Wall-clock = per-label Delta build + CP set construction.
        t0 = time.time()

        def delta_matrix(node_list):
            out = []
            for v in node_list:
                v = int(v)
                Sv = Sc[v * d:(v + 1) * d, :]
                out.append(label_deltas_for_node(
                    logits[v], Sv, W, kfac, LJ_fn(v), eps, float(u_all[v]), sk))
            return torch.stack(out, dim=0).to(dev)  # (n, C)

        d_cal = delta_matrix(cal_nodes)                        # (n_cal, C)
        d_sm = delta_matrix(sm_nodes)                          # (n_sm, C)  <-- same nodes
        cal_true = S_clean[cal_idx, y_sub[cal_idx]]
        d_cal_true = d_cal[torch.arange(len(cal_nodes)), y_sub[cal_idx]]
        q_rob = cp_quantile(cal_true - d_cal_true, alpha)
        # restrict conformal coverage/size to the SAME smoothing nodes for a fair head-to-head
        cov_conf, size_conf = coverage_and_size(
            S_clean[sm_node_set], y_sm, q_rob, delta_test=d_sm)
        wall_conf = time.time() - t0

        rows.append({
            "method": "AEGIS-Conformal", "matching": "-", "eps": eps,
            "coverage": cov_conf, "mean_set_size": size_conf, "n_nodes": n_sm,
            "wall_smoke_s": wall_conf, "samples_per_node": 0,
            "wall_extrap_1e4_s": wall_conf, "cert_frac": float("nan"), "sigma": 0.0,
        })

        # =========================== RANDOMIZED SMOOTHING ===========================
        # One smoothing run per sigma<->eps matching. The expensive M-sample
        # reconvergence is INDEPENDENT of the matching (noise scale only rescales
        # the samples), so we draw once at the largest sigma is NOT valid (variance
        # differs); we run each matching separately for soundness.
        for match in matchings:
            if match == "frob":
                sigma = eps / math.sqrt(2.0 * E)   # SAME Frobenius eps-ball as AEGIS
            elif match == "per_edge":
                sigma = eps                         # per-coordinate l2 std (larger ball)
            else:
                raise ValueError(f"unknown sigma-match {match}")
            t0 = time.time()
            n_cov = 0; size_sum = 0.0; n_cert = 0
            pA_hats = []; pA_los = []
            for v in sm_nodes:
                v = int(v); y_lab = int(y_sub[v])
                counts = smoothing_classify_node(
                    model, A_sub, ctx_sub, Z_sub, edge_idx, v, nc,
                    args.M, sigma, dev, args.smooth_tol, args.smooth_maxiter)
                cert = smoothed_certificate(counts, args.M, sigma, eps, alpha)
                if y_lab in cert["pred_set"]:
                    n_cov += 1
                size_sum += len(cert["pred_set"])
                n_cert += int(cert["certified"])
                pA_hats.append(cert["pA_hat"]); pA_los.append(cert["pA_lo"])
            wall_sm_smoke = time.time() - t0
            cov_sm = n_cov / n_sm
            size_sm = size_sum / n_sm
            cert_frac = n_cert / n_sm
            # extrapolate wall-clock to standard M=1e4 (linear in M; CP/Phi negligible)
            wall_sm_extrap = wall_sm_smoke * (args.extrap_M / args.M)

            rows.append({
                "method": "RandSmoothing", "matching": match, "eps": eps,
                "coverage": cov_sm, "mean_set_size": size_sm, "n_nodes": n_sm,
                "wall_smoke_s": wall_sm_smoke, "samples_per_node": args.M,
                "wall_extrap_1e4_s": wall_sm_extrap, "cert_frac": cert_frac,
                "sigma": sigma,
            })

            # ----------------------------- self-checks -----------------------------
            if abs(eps - 0.01) < 1e-12:
                selfcheck.append((f"smoothing[{match}]_cov>=1-alpha@eps0.01",
                                  cov_sm >= 1 - alpha - 1e-9,
                                  f"cov={cov_sm:.3f} target={1-alpha}"))
            selfcheck.append((f"smoothing[{match}]_wall>>conformal@eps{eps}",
                              wall_sm_smoke > 10.0 * max(wall_conf, 1e-6),
                              f"sm={wall_sm_smoke:.2f}s conf={wall_conf:.4f}s "
                              f"ratio={wall_sm_smoke/max(wall_conf,1e-9):.0f}x"))
            valid_p = all(0.0 <= p <= 1.0 for p in pA_hats) and \
                      all(lo <= ph + 1e-9 for lo, ph in zip(pA_los, pA_hats))
            selfcheck.append((f"smoothed[{match}]_p_valid@eps{eps}", valid_p,
                              f"pA_hat in [{min(pA_hats):.3f},{max(pA_hats):.3f}], "
                              f"cert_frac={cert_frac:.2f}"))

            print(f"  [eps={eps} smoothing/{match} sigma={sigma:.2e}] "
                  f"cov={cov_sm:.3f} size={size_sm:.2f} wall={wall_sm_smoke:.1f}s "
                  f"(M={args.M}) cert_frac={cert_frac:.2f} "
                  f"extrap1e4={wall_sm_extrap:.0f}s", flush=True)

        if abs(eps - 0.01) < 1e-12:
            selfcheck.append(("conformal_cov>=1-alpha@eps0.01",
                              cov_conf >= 1 - alpha - 1e-9,
                              f"cov={cov_conf:.3f} target={1-alpha}"))
        print(f"  [eps={eps} CONFORMAL] cov={cov_conf:.3f} size={size_conf:.2f} "
              f"wall={wall_conf:.4f}s (zero-sample)", flush=True)

    # --------------------------- write CSV ---------------------------
    out = RESULTS / args.out
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "method", "matching", "eps", "coverage", "mean_set_size", "n_nodes",
            "wall_smoke_s", "samples_per_node", "wall_extrap_1e4_s", "cert_frac", "sigma"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**r,
                         "coverage": round(r["coverage"], 4),
                         "mean_set_size": round(r["mean_set_size"], 3),
                         "wall_smoke_s": round(r["wall_smoke_s"], 4),
                         "wall_extrap_1e4_s": round(r["wall_extrap_1e4_s"], 1),
                         "cert_frac": (round(r["cert_frac"], 3)
                                       if r["cert_frac"] == r["cert_frac"] else ""),
                         "sigma": f"{r['sigma']:.3e}"})

    # --------------------------- compact table ---------------------------
    print(f"\n{'='*98}")
    print(f"=== CONFORMAL vs SMOOTHING — {args.dataset} seed {args.seed}, "
          f"alpha={alpha} (target cov {1-alpha}), score={sk}, n_nodes={n_sm} ===")
    print(f"  sigma<->eps matchings: frob = sigma=eps/sqrt(2|E|) (SAME Frobenius eps-ball, "
          f"|E|={E}); per_edge = sigma=eps (larger per-coordinate ball, favorable to smoothing)")
    hdr = f"  {'method':<16}{'match':>9}{'eps':>6}{'coverage':>10}{'set_size':>10}" \
          f"{'cert_frac':>10}{'wall_smoke':>12}{'M/node':>8}{'wall@1e4':>13}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        cf = f"{r['cert_frac']:.2f}" if r["cert_frac"] == r["cert_frac"] else "  -"
        print(f"  {r['method']:<16}{r['matching']:>9}{r['eps']:>6}{r['coverage']:>10.3f}"
              f"{r['mean_set_size']:>10.2f}{cf:>10}{r['wall_smoke_s']:>10.3f}s"
              f"{r['samples_per_node']:>8}{r['wall_extrap_1e4_s']:>11.1f}s")

    # winners per metric: Conformal vs EACH smoothing matching
    print(f"\n  WINNERS (per eps, Conformal vs each smoothing matching):")
    for eps in eps_list:
        rc = next(r for r in rows if r["method"] == "AEGIS-Conformal" and r["eps"] == eps)
        for match in matchings:
            rs = next(r for r in rows if r["method"] == "RandSmoothing"
                      and r["eps"] == eps and r["matching"] == match)
            cov_win = "Conformal" if rc["coverage"] > rs["coverage"] + 1e-9 else (
                "Smoothing" if rs["coverage"] > rc["coverage"] + 1e-9 else "tie")
            size_win = "Conformal" if rc["mean_set_size"] < rs["mean_set_size"] - 1e-9 else (
                "Smoothing" if rs["mean_set_size"] < rc["mean_set_size"] - 1e-9 else "tie")
            speedup = rs["wall_extrap_1e4_s"] / max(rc["wall_extrap_1e4_s"], 1e-9)
            wc_win = f"Conformal ({speedup:.0f}x faster @1e4)"
            print(f"   eps={eps} vs smoothing[{match}]: coverage->{cov_win}  "
                  f"set_size->{size_win}  wall-clock->{wc_win}")

    # self-check report
    print(f"\n  SELF-CHECKS:")
    all_ok = True
    for name, ok, detail in selfcheck:
        all_ok &= ok
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n  {'ALL SELF-CHECKS PASSED' if all_ok else '*** SOME SELF-CHECKS FAILED ***'}")
    print(f"  total wall: {time.time()-t_all:.0f}s   wrote: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
