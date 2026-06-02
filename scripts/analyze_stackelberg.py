"""Aggregate the AEGIS-Stackelberg coverage CSVs into the findings tables.

Reads results/stackelberg_coverage_{residual,ceiling,submod,damage}.csv and prints
(to stdout) compact mean +/- sd tables + verdicts:
  - residual-vs-B per method per dataset (+ reduction %)
  - DOMINANCE: portfolio(best r) vs v_ij / each centrality null / random
  - CEILING: residual vs sigma_{r+1}; fraction of (seed,B) with residual>=sigma_{r+1}
  - SUBMODULAR vs MODULAR: greedy - top-B gap, set-equality fraction
  - DAMAGE: reconverged ||z*(A+d)-z*(A)|| per method at B=DAMAGE_B

Usage: .venv/bin/python scripts/analyze_stackelberg.py
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

R = Path("results")


def load(name):
    p = R / f"stackelberg_coverage_{name}.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return math.nan


def ms(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return math.nan, math.nan, 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def main():
    res = load("residual"); ceil = load("ceiling"); sub = load("submod"); dmg = load("damage")
    datasets = []
    for r in res:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    print("#" * 78)
    print("AEGIS-STACKELBERG COVERAGE -- AGGREGATED FINDINGS")
    print("#" * 78)

    # ---------------- (1) residual-vs-B per method/dataset ----------------
    for ds in datasets:
        rows = [r for r in res if r["dataset"] == ds]
        budgets = sorted({int(r["B"]) for r in rows})
        methods = []
        for r in rows:
            if r["method"] not in methods:
                methods.append(r["method"])
        reduced = any(int(r.get("reduced", 0)) for r in rows)
        sig1 = ms([fnum(r["sigma1_clean"]) for r in rows])[0]
        print(f"\n===== {ds}  (sigma1_clean~{sig1:.4f}{'  [REDUCED protocol]' if reduced else ''}) =====")
        hdr = "method".ljust(16) + "".join(f"B={b}".rjust(13) for b in budgets)
        print(hdr); print("-" * len(hdr))
        # order: portfolios, vij, greedy, centralities, random
        def order_key(m):
            pref = {"portfolio_r1": 0, "portfolio_r5": 1, "portfolio_r10": 2,
                    "vij": 3, "greedy": 4}
            if m in pref: return (pref[m], m)
            if m.startswith("cent_"): return (5, m)
            if m == "random": return (6, m)
            return (7, m)
        for m in sorted(methods, key=order_key):
            cells = []
            for b in budgets:
                vals = [fnum(r["residual"]) for r in rows if r["method"] == m and int(r["B"]) == b]
                mm, sd, n = ms(vals)
                if math.isnan(mm):
                    cells.append("n/a".rjust(13))
                else:
                    red = 100.0 * (1 - mm / sig1) if sig1 else 0.0
                    cells.append(f"{mm:.3f}({red:+.1f}%)".rjust(13))
            print(m.ljust(16) + "".join(cells))

    # ---------------- (2) DOMINANCE ----------------
    print("\n" + "#" * 78)
    print("(2) DOMINANCE: does portfolio(best r) give the LOWEST residual?")
    print("    margin = (residual_other - residual_portfolio) ; >0 => portfolio wins")
    print("#" * 78)
    for ds in datasets:
        rows = [r for r in res if r["dataset"] == ds]
        budgets = sorted({int(r["B"]) for r in rows})
        port_methods = sorted({r["method"] for r in rows if r["method"].startswith("portfolio_r")})
        # best portfolio = lowest mean residual at the largest budget
        Bmax = max(budgets)
        best_port, best_val = None, math.inf
        for pm in port_methods:
            mm = ms([fnum(r["residual"]) for r in rows if r["method"] == pm and int(r["B"]) == Bmax])[0]
            if not math.isnan(mm) and mm < best_val:
                best_val, best_port = mm, pm
        print(f"\n----- {ds}  (best portfolio = {best_port}) -----")
        rivals = [m for m in {r["method"] for r in rows}
                  if m != best_port and not m.startswith("portfolio_r")]
        def rk(m):
            return ({"vij":0,"greedy":1}.get(m, 5 if m.startswith("cent_") else (6 if m=="random" else 7)), m)
        for b in budgets:
            pv = ms([fnum(r["residual"]) for r in rows if r["method"]==best_port and int(r["B"])==b])[0]
            parts = []
            for m in sorted(rivals, key=rk):
                ov = ms([fnum(r["residual"]) for r in rows if r["method"]==m and int(r["B"])==b])[0]
                if math.isnan(ov) or math.isnan(pv):
                    parts.append(f"{m}:n/a")
                else:
                    parts.append(f"{m}:{ov-pv:+.4f}")
            print(f"  B={b:3d}  portfolio={pv:.4f} | margins  " + "  ".join(parts))
        # verdict: portfolio strictly <= every non-portfolio rival at all B (within seeds)
        wins = True
        for b in budgets:
            pv = ms([fnum(r["residual"]) for r in rows if r["method"]==best_port and int(r["B"])==b])[0]
            for m in rivals:
                ov = ms([fnum(r["residual"]) for r in rows if r["method"]==m and int(r["B"])==b])[0]
                if not math.isnan(ov) and ov < pv - 1e-9:
                    wins = False
        print(f"  VERDICT[{ds}]: portfolio dominates all nulls/random at every B? -> {wins}")

    # ---------------- (3) CEILING ----------------
    print("\n" + "#" * 78)
    print("(3) CERTIFIED CEILING: residual(portfolio-r) vs sigma_{r+1}(S_c)")
    print("    holds = residual >= sigma_{r+1} (clean interlacing lower bound)")
    print("#" * 78)
    for ds in datasets:
        rows = [r for r in ceil if r["dataset"] == ds and int(r["r"]) >= 0]
        if not rows: continue
        rset = sorted({int(r["r"]) for r in rows})
        budgets = sorted({int(r["B"]) for r in rows})
        print(f"\n----- {ds} -----")
        for r in rset:
            sigrp1 = ms([fnum(x["sigma_r_plus_1"]) for x in rows if int(x["r"])==r])[0]
            line = f"  r={r:2d} (sigma_{r+1}={sigrp1:.4f}): "
            holds_all = []
            for b in budgets:
                sub_b = [x for x in rows if int(x["r"])==r and int(x["B"])==b]
                resb = ms([fnum(x["residual"]) for x in sub_b])[0]
                hold_frac = sum(int(x["holds"]) for x in sub_b)/max(len(sub_b),1)
                holds_all.append(hold_frac)
                line += f"B{b}:{resb:.4f}/{hold_frac:.0%} "
            print(line)
        # overall: does residual TRACK sigma_{r+1} (close) and is the bound respected?
        gaps = [fnum(x["gap"]) for x in rows]
        gm, gsd, _ = ms(gaps)
        held = sum(int(x["holds"]) for x in rows)/max(len(rows),1)
        print(f"  -> mean(residual - sigma_(r+1)) = {gm:+.5f} +/- {gsd:.5f} ; bound held in {held:.0%} of cases")

    # ---------------- (4) SUBMODULAR vs MODULAR ----------------
    print("\n" + "#" * 78)
    print("(4) SUBMODULAR vs MODULAR: top-B-by-energy vs GREEDY-COVERAGE")
    print("    gap = residual_greedy - residual_topB (<0 => greedy strictly better => submodular)")
    print("#" * 78)
    if not sub:
        print("  (no submod rows -- greedy disabled for all datasets run)")
    for ds in sorted({r["dataset"] for r in sub}):
        rows = [r for r in sub if r["dataset"] == ds]
        rset = sorted({int(r["r"]) for r in rows})
        budgets = sorted({int(r["B"]) for r in rows})
        print(f"\n----- {ds} -----")
        for r in rset:
            line = f"  r={r:2d}: "
            for b in budgets:
                sb = [x for x in rows if int(x["r"])==r and int(x["B"])==b]
                gap = ms([fnum(x["greedy_minus_topB"]) for x in sb])[0]
                eq = ms([float(x["set_equal"]) for x in sb])[0]
                line += f"B{b}:gap={gap:+.5f}(eq={eq:.0%}) "
            print(line)
        allgap = [fnum(x["greedy_minus_topB"]) for x in rows]
        gm, gsd, _ = ms(allgap)
        eqf = ms([float(x["set_equal"]) for x in rows])[0]
        # verdict
        if gm < -1e-4:
            v = "SUBMODULAR (greedy strictly beats top-B; (1-1/e) story applies)"
        elif abs(gm) <= 1e-4 and eqf > 0.5:
            v = "MODULAR (top-B == greedy; top-B-by-energy is exactly optimal)"
        else:
            v = "EFFECTIVELY MODULAR (greedy ~ top-B within noise; no meaningful submodular gain)"
        print(f"  -> mean gap(greedy-topB) = {gm:+.6f} +/- {gsd:.6f} ; set-equal {eqf:.0%}")
        print(f"  VERDICT[{ds}]: {v}")

    # ---------------- (5) DAMAGE CONFIRMATION ----------------
    print("\n" + "#" * 78)
    print("(5) DAMAGE CONFIRMATION at B=20: reconverged ||z*(A+delta)-z*(A)||")
    print("    attack = leading right singular vec of masked S_c, ||delta||_F=0.10")
    print("#" * 78)
    for ds in sorted({r["dataset"] for r in dmg}):
        rows = [r for r in dmg if r["dataset"] == ds]
        methods = sorted({r["method"] for r in rows},
                         key=lambda m:(0 if m=="none" else 1 if m.startswith("portfolio") else
                                       2 if m=="vij" else 3 if m=="greedy" else
                                       5 if m.startswith("cent_") else 7, m))
        none_d = ms([fnum(r["damage"]) for r in rows if r["method"]=="none"])[0]
        print(f"\n----- {ds}  (no-defense damage = {none_d:.4f}) -----")
        for m in methods:
            d = ms([fnum(r["damage"]) for r in rows if r["method"]==m])
            s = ms([fnum(r["residual_sigma1"]) for r in rows if r["method"]==m])[0]
            red = 100.0*(1 - d[0]/none_d) if (none_d and not math.isnan(d[0])) else math.nan
            print(f"  {m.ljust(16)} damage={d[0]:.4f}+/-{d[1]:.4f}  sigma1={s:.4f}  vs-none={red:+.1f}%")
        # does the best portfolio reduce damage vs none AND vs every centrality null?
        port = [m for m in methods if m.startswith("portfolio")]
        cents = [m for m in methods if m.startswith("cent_")]
        if port:
            pd = ms([fnum(r["damage"]) for r in rows if r["method"]==port[0]])[0]
            beats_none = pd < none_d - 1e-6
            beats_cent = all(pd <= ms([fnum(r["damage"]) for r in rows if r["method"]==c])[0] + 1e-6
                             for c in cents) if cents else None
            print(f"  VERDICT[{ds}]: portfolio damage < none? {beats_none} ; <= all centrality nulls? {beats_cent}")

    print("\n" + "#" * 78)
    print("END")


if __name__ == "__main__":
    main()
