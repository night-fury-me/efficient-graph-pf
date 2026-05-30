"""Revision-R2 R5 — PTDF (standalone, no outage correction) baseline.

For each line k = (b_from, b_to), compute a DC-PTDF-only screening score:
    score_ptdf(k) = sum_l |PTDF_{l, b_from} - PTDF_{l, b_to}|
where PTDF is the DC sensitivity matrix from the B-matrix.  This is what
LODF reduces to BEFORE the outage-correction (1 - PTDF_{k,k}) division;
the comparison answers reviewer R5 (does PTDF alone beat AEGIS on N-1?).

Closes: R5 in docs/review_2026-05-29_r2/06_editorial_decision.md.

Usage: .venv/bin/python scripts/revision_R2/R2_12_ptdf_baseline.py
"""
from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

try:
    import pandapower as pp
    import pandapower.networks as pp_nets
except ImportError as exc:
    sys.exit(f"pandapower required ({exc})")

OUT_CSV = Path("results/revision_R2/ptdf_baseline.csv")

CASES = {"case57": pp_nets.case57, "case118": pp_nets.case118}
SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def compute_ptdf(net):
    """DC PTDF matrix [n_lines, n_buses] via B-matrix.

    Uses pandapower's makeBdc on the converged DC base-case.  Returns a dense
    PTDF where row l, col b = partial F_l / partial P_b at slack-out reference.
    """
    # Pandapower internal: pp.pypower or pp.makeBdc not always exposed; use
    # the standard interpretation via line susceptances + bus injection inversion.
    try:
        from pandapower.pypower.makeBdc import makeBdc
        from pandapower.pd2ppc import _pd2ppc
    except ImportError:
        from pandapower.pypower.makeBdc import makeBdc
        from pandapower.pd2ppc import _pd2ppc
    # Convert
    _pd2ppc(net)
    ppc = net["_ppc"]
    baseMVA = ppc["baseMVA"]
    bus = ppc["bus"]
    branch = ppc["branch"]
    n_bus = bus.shape[0]
    n_br = branch.shape[0]
    # Slack identification
    REF = 3
    ref_mask = (bus[:, 1] == REF)
    if not ref_mask.any():
        ref_idx = 0
    else:
        ref_idx = int(np.where(ref_mask)[0][0])
    noref = np.array([i for i in range(n_bus) if i != ref_idx])
    res = makeBdc(bus, branch)
    # newer pandapower returns 6 values (Bbus, Bf, Pbusinj, Pfinj, Cft, Yfdc) or 4
    Bbus, Bf = res[0], res[1]
    # Solve B_nn * theta = P_n for unit injections; PTDF = Bf * B^-1
    Bbus_d = Bbus.toarray() if hasattr(Bbus, "toarray") else np.asarray(Bbus)
    Bf_d = Bf.toarray() if hasattr(Bf, "toarray") else np.asarray(Bf)
    # Reduced B (drop slack row/col)
    Bnn = Bbus_d[np.ix_(noref, noref)]
    inv = np.linalg.inv(Bnn)
    # Expand to full
    PTDF = np.zeros((n_br, n_bus))
    PTDF[:, noref] = Bf_d[:, noref] @ inv
    return PTDF, ref_idx


def line_endpoints(net):
    """Return [(b_from, b_to)] in line index order, mapped to ppc-indexed buses."""
    bus_lookup = {b: i for i, b in enumerate(net.bus.index)}
    return [(bus_lookup[int(r["from_bus"])], bus_lookup[int(r["to_bus"])])
            for _, r in net.line.iterrows()]


def ptdf_score(PTDF, endpoints):
    """For each line k = (i, j), score = sum_l |PTDF[l, i] - PTDF[l, j]|.

    This is the unsigned redistribution magnitude without outage correction.
    """
    scores = []
    for (i, j) in endpoints:
        col = PTDF[:, i] - PTDF[:, j]
        scores.append(float(np.sum(np.abs(col))))
    return np.array(scores)


def true_n1_severity(net, line_idx):
    """AC N-1 severity = L2(Vp - Vb, theta_p - theta_b)."""
    net2 = copy.deepcopy(net)
    pp.runpp(net2)
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
        net = case_fn()
        pp.rundcpp(net)
        for seed in SEEDS:
            np.random.seed(seed)
            net_seed = copy.deepcopy(net)
            noise = 1.0 + 0.05 * (2 * np.random.rand(len(net_seed.load)) - 1)
            net_seed.load["p_mw"] = net_seed.load["p_mw"] * noise
            pp.rundcpp(net_seed)
            PTDF, _ = compute_ptdf(net_seed)
            endpoints = line_endpoints(net_seed)
            ptdf_s = ptdf_score(PTDF, endpoints)
            true_s = np.array([true_n1_severity(net_seed, li)
                               for li in net_seed.line.index])
            mask = np.isfinite(ptdf_s) & np.isfinite(true_s)
            if mask.sum() < 3:
                continue
            tau, p_tau = kendalltau(ptdf_s[mask], true_s[mask])
            top10_p = set(np.argsort(-ptdf_s[mask])[:10])
            top10_t = set(np.argsort(-true_s[mask])[:10])
            p_at_10 = len(top10_p & top10_t) / 10.0
            rows.append({"case": cname, "seed": seed,
                         "n_lines": int(mask.sum()),
                         "tau_ptdf_vs_true_n1": float(tau),
                         "p_at_10_ptdf": p_at_10,
                         "ptdf_tau_pvalue": float(p_tau)})
            print(f"  {cname:8s} seed={seed:5d} "
                  f"tau_PTDF={tau:+.3f} P@10={p_at_10:.2f}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    print("\nPer-case PTDF aggregates (across 10 noisy-load seeds):")
    for cname in CASES:
        sub = [r for r in rows if r["case"] == cname]
        if not sub:
            continue
        taus = np.array([r["tau_ptdf_vs_true_n1"] for r in sub])
        pats = np.array([r["p_at_10_ptdf"] for r in sub])
        print(f"  {cname:8s}  PTDF tau={taus.mean():+.3f}±{taus.std():.3f}  "
              f"P@10={pats.mean():.2f}±{pats.std():.2f}")


if __name__ == "__main__":
    main()
