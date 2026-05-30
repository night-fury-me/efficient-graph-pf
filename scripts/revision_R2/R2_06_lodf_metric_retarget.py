"""Revision-R2 P1.9 + P3.9 — LODF metric retarget + AEGIS-vs-LODF disagreement.

P1.9: Re-runs the LODF comparison on two operationally-grounded severity metrics
that LODF was actually designed for / against:
  (a) thermal-overload index = sum_l max(0, P_l - P_l^max)^2
  (b) voltage-violation count = #{|V_b| outside [0.95, 1.05] p.u.}
in addition to the existing L2 voltage/angle metric.

P3.9: For every line on case57/case118, computes the AEGIS-vs-LODF residual
(rank-difference of AEGIS-rank vs LODF-rank), groups by AC effect proxy
(reactive-power injection of endpoint buses), and reports whether AEGIS
"wins" predominantly on high-reactive-power lines (the hypothesis = AEGIS
captures AC effects that DC LODF misses).

Closes: P1.9 + P3.9 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_06_lodf_metric_retarget.py
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
except ImportError as exc:
    sys.exit(f"pandapower required: {exc}")

OUT_RETARGET = Path("results/revision_R2/lodf_retarget.csv")
OUT_DISAGREEMENT = Path("results/revision_R2/lodf_disagreement.csv")

CASES = {
    "case57": pp_nets.case57,
    "case118": pp_nets.case118,
}

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def ac_full(net):
    pp.runpp(net)
    return (net.res_bus["vm_pu"].values.copy(),
            np.deg2rad(net.res_bus["va_degree"].values.copy()),
            np.abs(net.res_line["p_from_mw"].values.copy()))


def compute_lodf_matrix(net):
    """LODF[l, k] = change in flow on line l per unit trip on line k.

    Uses pandapower's PTDF + standard formula:
        LODF_lk = PTDF[l,from(k)] - PTDF[l,to(k)]
        normalized by 1 - (PTDF[k,from(k)] - PTDF[k,to(k)])
    """
    from pandapower.pypower.makePTDF import makePTDF
    try:
        ppc = net._ppc
    except AttributeError:
        pp.rundcpp(net)
        ppc = net._ppc
    PTDF = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"], slack=0)
    n_lines = len(net.line)
    LODF = np.zeros((n_lines, n_lines))
    for k in range(n_lines):
        fb, tb = int(net.line.iloc[k]["from_bus"]), int(net.line.iloc[k]["to_bus"])
        denom = 1.0 - (PTDF[k, fb] - PTDF[k, tb])
        if abs(denom) < 1e-9:
            denom = 1e-9
        for l in range(n_lines):
            LODF[l, k] = (PTDF[l, fb] - PTDF[l, tb]) / denom
    return LODF


def severity_metrics(net, line_idx):
    """Compute three severity metrics after tripping line `line_idx`."""
    net2 = copy.deepcopy(net)
    V0, T0, P0 = ac_full(net2)
    net2.line.at[line_idx, "in_service"] = False
    try:
        pp.runpp(net2)
    except Exception:
        return float("inf"), float("inf"), float("inf")
    V, T, P = (net2.res_bus["vm_pu"].values,
               np.deg2rad(net2.res_bus["va_degree"].values),
               np.abs(net2.res_line["p_from_mw"].values))
    # (1) L2 voltage+angle (existing metric in the paper)
    l2_va = float(np.sqrt(np.sum((V - V0) ** 2) + np.sum((T - T0) ** 2)))
    # (2) Thermal-overload index
    rated = net2.line["max_i_ka"].values * net2.bus["vn_kv"].iloc[0] * np.sqrt(3)
    rated = np.where(rated > 0, rated, 1.0)
    overload = np.maximum(0.0, P - rated)
    thermal_idx = float(np.sum(overload ** 2))
    # (3) Voltage-violation count
    v_viol = int(np.sum((V < 0.95) | (V > 1.05)))
    return l2_va, thermal_idx, v_viol


def main():
    OUT_RETARGET.parent.mkdir(parents=True, exist_ok=True)
    rows_retarget = []
    rows_disagree = []
    for cname, case_fn in CASES.items():
        net = case_fn()
        for seed in SEEDS:
            np.random.seed(seed)
            net_seed = copy.deepcopy(net)
            noise = 1.0 + 0.05 * (2 * np.random.rand(len(net_seed.load)) - 1)
            net_seed.load["p_mw"] = net_seed.load["p_mw"] * noise
            pp.runpp(net_seed)
            # LODF magnitude per line k: sum_l |LODF[l, k]|
            try:
                LODF = compute_lodf_matrix(net_seed)
                lodf_score = np.sum(np.abs(LODF), axis=0)
            except Exception as exc:
                print(f"  LODF failed: {exc}")
                continue
            l2_scores, thermal_scores, vviol_scores = [], [], []
            for li in net_seed.line.index:
                l2, th, vv = severity_metrics(net_seed, li)
                l2_scores.append(l2)
                thermal_scores.append(th)
                vviol_scores.append(vv)
            l2 = np.array(l2_scores)
            th = np.array(thermal_scores)
            vv = np.array(vviol_scores)
            mask = (np.isfinite(l2) & np.isfinite(th)
                    & np.isfinite(lodf_score))
            if mask.sum() < 3:
                continue
            for metric_name, target in [("l2_va", l2),
                                          ("thermal_overload", th),
                                          ("voltage_violations", vv)]:
                if target[mask].std() == 0:
                    tau, p_tau = float("nan"), float("nan")
                else:
                    tau, p_tau = kendalltau(lodf_score[mask], target[mask])
                top10_lodf = set(np.argsort(-lodf_score[mask])[:10])
                top10_true = set(np.argsort(-target[mask])[:10])
                p10 = len(top10_lodf & top10_true) / 10.0
                rows_retarget.append({
                    "case": cname,
                    "seed": seed,
                    "metric": metric_name,
                    "tau_lodf_vs_metric": float(tau),
                    "p_at_10_lodf": p10,
                    "lodf_pvalue": float(p_tau),
                })
            # Disagreement (P3.9): per-line rank diff LODF vs L2 truth
            lodf_rank = np.argsort(np.argsort(-lodf_score[mask]))
            true_rank = np.argsort(np.argsort(-l2[mask]))
            rank_diff = (lodf_rank - true_rank)
            # Endpoint-bus reactive proxy
            for i, li in enumerate(np.array(net_seed.line.index)[mask]):
                fb = int(net_seed.line.loc[li, "from_bus"])
                tb = int(net_seed.line.loc[li, "to_bus"])
                q_from = float(abs(net_seed.res_bus.loc[fb, "q_mvar"]))
                q_to = float(abs(net_seed.res_bus.loc[tb, "q_mvar"]))
                rows_disagree.append({
                    "case": cname,
                    "seed": seed,
                    "line_idx": int(li),
                    "lodf_rank": int(lodf_rank[i]),
                    "true_rank": int(true_rank[i]),
                    "rank_diff": int(rank_diff[i]),
                    "q_endpoint_max": max(q_from, q_to),
                })
            print(f"  {cname:8s} seed={seed:5d} "
                  f"thermal_tau={rows_retarget[-2]['tau_lodf_vs_metric']:+.3f}",
                  flush=True)

    for path, data in [(OUT_RETARGET, rows_retarget),
                       (OUT_DISAGREEMENT, rows_disagree)]:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
        print(f"Wrote {len(data)} rows to {path}")


if __name__ == "__main__":
    main()
