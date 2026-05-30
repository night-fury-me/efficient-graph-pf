"""Revision-R2 P1.8 — Performance-Index (Ejebe-Wollenberg) ranking baseline.

Computes the classical PI = sum_l ((P_l^post / P_l^rated)^{2n}) on case57 and
case118 brute-force N-1 contingencies (n = 2), and reports Kendall tau and
P@10 against the brute-force N-1 severity ranking, alongside the existing LODF
and AEGIS rankings.

PI definition (Wood-Wollenberg Ch. 11, Ejebe 1979 IEEE TPAS):
    PI(k) = sum_l |P_l^k / P_l^max|^{2n}
where P_l^k is the post-contingency-k line flow on line l, P_l^max is the
thermal limit, and n is the order (we use n = 2 — quadratic-line-loading PI).

Closes: P1.8 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_05_pi_baseline.py
"""
from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wilcoxon

try:
    import pandapower as pp
    import pandapower.networks as pp_nets
    import pandapower.contingency as pp_cont
except ImportError as exc:
    sys.exit(f"pandapower required: pip install pandapower  ({exc})")

OUT_CSV = Path("results/revision_R2/pi_baseline.csv")

CASES = {
    "case57": pp_nets.case57,
    "case118": pp_nets.case118,
}

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def load_case(case_fn):
    net = case_fn()
    # Run DC power flow to obtain base-case flows
    pp.rundcpp(net)
    return net


def pi_index(net, line_idx, n_power=2):
    """Compute PI = sum_l (P_l / P_max_l)^{2n} after tripping line `line_idx`."""
    net2 = copy.deepcopy(net)
    # Disable line
    net2.line.at[line_idx, "in_service"] = False
    try:
        pp.rundcpp(net2)
    except (pp.LoadflowNotConverged, Exception):
        return float("inf")  # Diverged contingency = severe
    flows = np.abs(net2.res_line["p_from_mw"].values)
    rated = net2.line["max_i_ka"].values * net2.bus["vn_kv"].iloc[0] * np.sqrt(3)
    rated = np.where(rated > 0, rated, 1.0)
    ratio = flows / rated
    return float(np.sum(ratio ** (2 * n_power)))


def true_n1_severity(net, line_idx):
    """Brute-force severity = post-contingency L2 norm of voltage+angle delta."""
    net2 = copy.deepcopy(net)
    pp.runpp(net2)  # AC base
    Vb = net2.res_bus["vm_pu"].values.copy()
    Tb = np.deg2rad(net2.res_bus["va_degree"].values.copy())
    net2.line.at[line_idx, "in_service"] = False
    try:
        pp.runpp(net2)
    except Exception:
        return float("inf")
    Vp = net2.res_bus["vm_pu"].values
    Tp = np.deg2rad(net2.res_bus["va_degree"].values)
    return float(np.sqrt(np.sum((Vp - Vb) ** 2) + np.sum((Tp - Tb) ** 2)))


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cname, case_fn in CASES.items():
        net = load_case(case_fn)
        # PI is deterministic given a network -> no seed loop needed for PI itself,
        # but we replicate with synthetic load noise for confidence intervals.
        for seed in SEEDS:
            np.random.seed(seed)
            net_seed = copy.deepcopy(net)
            # Inject mild load noise (+- 5%) to break PI-LODF degeneracy
            noise = 1.0 + 0.05 * (2 * np.random.rand(len(net_seed.load)) - 1)
            net_seed.load["p_mw"] = net_seed.load["p_mw"] * noise
            pp.rundcpp(net_seed)
            pi_scores = []
            true_scores = []
            for li in net_seed.line.index:
                pi_scores.append(pi_index(net_seed, li))
                true_scores.append(true_n1_severity(net_seed, li))
            pi_scores = np.array(pi_scores)
            true_scores = np.array(true_scores)
            # Filter out diverged
            mask = np.isfinite(pi_scores) & np.isfinite(true_scores)
            if mask.sum() < 3:
                continue
            tau, p_tau = kendalltau(pi_scores[mask], true_scores[mask])
            # P@10
            top10_pi = set(np.argsort(-pi_scores[mask])[:10])
            top10_true = set(np.argsort(-true_scores[mask])[:10])
            p_at_10 = len(top10_pi & top10_true) / 10.0
            rows.append({
                "case": cname,
                "seed": seed,
                "n_lines": int(mask.sum()),
                "tau_pi_vs_true_n1": float(tau),
                "p_at_10_pi": p_at_10,
                "pi_tau_pvalue": float(p_tau),
            })
            print(f"  {cname:8s} seed={seed:5d} tau_PI={tau:+.3f} "
                  f"P@10_PI={p_at_10:.2f}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # Aggregate
    print("\nPer-case PI aggregates (across 10 noisy-load seeds):")
    for cname in CASES:
        sub = [r for r in rows if r["case"] == cname]
        if not sub:
            continue
        taus = np.array([r["tau_pi_vs_true_n1"] for r in sub])
        pats = np.array([r["p_at_10_pi"] for r in sub])
        print(f"  {cname:8s}  PI tau={taus.mean():+.3f}±{taus.std():.3f}  "
              f"P@10={pats.mean():.2f}±{pats.std():.2f}")
    print("\nReport these alongside AEGIS/LODF tau in case_study.tex.")


if __name__ == "__main__":
    main()
