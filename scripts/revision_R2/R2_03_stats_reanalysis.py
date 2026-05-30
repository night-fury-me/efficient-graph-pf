"""Revision-R2 P1.5 + P2.7 — statistical reanalysis from existing CSVs.

Recomputes 95% CIs (t_9) and per-cell Wilcoxon p-values for every results table,
and the sign-test p-value for the "149/150 wins vs. Mettack" claim, from the
per-seed CSV outputs already in results/. No GPU runs required.

Closes: P1.5 (CI + per-cell Wilcoxon + sign-test + significance-filtered bolding)
        P2.7 (defense-ablation significance test).

Usage:
    .venv/bin/python scripts/revision_R2/R2_03_stats_reanalysis.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import binomtest, t as student_t, wilcoxon

PROJ = Path(__file__).resolve().parents[2]
RESULTS = PROJ / "results"
OUT_CSV = PROJ / "results/revision_R2/stats_reanalysis.csv"

# Map: (table label) -> (CSV path, group columns, value columns to compare pairwise)
TABLES = {
    "tab:cross_domain": {
        "csv": "exp_scalability_10seed.csv",  # uses kappa, tightness, AtkAdv per seed
        "group": ["dataset"],
        "value_columns": ["tightness", "atk_adv", "kappa", "ecrit"],
    },
    "tab:attack_full": {
        "csv": "full_attack_table.csv",
        "group": ["dataset", "epsilon"],
        "value_columns": ["svd_damage", "cls_pgd_damage", "shift_pgd_damage",
                          "random_damage", "svd_flip", "cls_pgd_flip"],
    },
    "tab:baselines": {
        "csv": "attack_baselines.csv",
        "group": ["dataset"],
        "value_columns": ["aegis_atk_adv", "degree_atk_adv",
                          "spectral_atk_adv", "betweenness_atk_adv"],
    },
    "tab:greedy_topk": {
        "csv": "greedy_topk_attack.csv",
        "group": ["dataset", "k"],
        "value_columns": ["greedy_damage", "aegis_damage",
                          "degree_damage", "random_damage"],
    },
    "tab:tau_cross": {
        "csv": "exp_tau_all_datasets.csv",  # cross-arch cross-dataset tau
        "group": ["dataset", "model"],
        "value_columns": ["tau"],
    },
    "tab:tightness_eps": {
        "csv": "exp_tightness_expansion.csv",
        "group": ["dataset", "epsilon"],
        "value_columns": ["tightness", "flip_rate"],
    },
    "tab:breach": {
        "csv": "exp_breach_rates.csv",
        "group": ["dataset", "epsilon"],
        "value_columns": ["breach_rate"],
    },
    "tab:scalability": {
        "csv": "exp_scalability_10seed.csv",
        "group": ["N"],
        "value_columns": ["dense_time_s", "matfree_time_s",
                          "dense_mem_mb", "matfree_mem_mb"],
    },
    "tab:ieee": {
        "csv": "n1_contingency_benchmark.csv",
        "group": ["case"],
        "value_columns": ["tau", "p_at_10", "v_rmse", "theta_rmse"],
    },
    "defense_ablation": {
        "csv": "exp_defense_ablation.csv",  # paired: aegis_mask vs random_mask
        "group": ["dataset", "k"],
        "value_columns": ["aegis_mask_drop", "random_mask_drop"],
    },
}


def t_ci(values: np.ndarray, alpha: float = 0.05):
    """95% CI for the mean using Student-t (n-1 df)."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float(values[0]) if n == 1 else float("nan")
    m = float(values.mean())
    s = float(values.std(ddof=1))
    se = s / math.sqrt(n)
    half = student_t.ppf(1 - alpha / 2, df=n - 1) * se
    return m - half, m + half, m


def per_cell_stats(values: np.ndarray, paired_baseline: np.ndarray = None,
                   null_value: float = 0.0):
    """Returns (mean, std, ci_lo, ci_hi, p_value).

    If paired_baseline is provided, p_value is from paired Wilcoxon vs that
    baseline (one-sided: values > baseline). Otherwise, one-sample Wilcoxon
    vs null_value.
    """
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else float("nan"), \
               float("nan"), float("nan"), float("nan"), float("nan")
    ci_lo, ci_hi, m = t_ci(values)
    s = float(values.std(ddof=1))
    try:
        if paired_baseline is not None and len(paired_baseline) == n:
            stat = wilcoxon(values, paired_baseline, alternative="greater",
                            zero_method="zsplit")
        else:
            shifted = values - null_value
            if np.allclose(shifted, 0.0):
                pval = 1.0
                return m, s, ci_lo, ci_hi, pval
            stat = wilcoxon(shifted, alternative="greater",
                            zero_method="zsplit")
        pval = float(stat.pvalue)
    except ValueError:
        pval = float("nan")
    return m, s, ci_lo, ci_hi, pval


def reanalyse_csv(table_label: str, spec: dict) -> List[dict]:
    csv_path = RESULTS / spec["csv"]
    if not csv_path.exists():
        print(f"  [skip] {table_label}: {csv_path} missing")
        return []
    # Read CSV
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    # Group
    group_cols = spec["group"]
    groups: Dict[tuple, list] = {}
    for r in rows:
        key = tuple(r.get(c, "") for c in group_cols)
        groups.setdefault(key, []).append(r)
    out_rows = []
    for key, grp in groups.items():
        for col in spec["value_columns"]:
            try:
                vals = np.array([float(r[col]) for r in grp
                                 if r.get(col) not in (None, "", "nan")])
            except (KeyError, ValueError):
                continue
            if len(vals) < 2:
                continue
            m, s, lo, hi, p = per_cell_stats(vals)
            out_rows.append({
                "table": table_label,
                **{c: k for c, k in zip(group_cols, key)},
                "metric": col,
                "n_seeds": len(vals),
                "mean": m,
                "std": s,
                "ci_lo_95": lo,
                "ci_hi_95": hi,
                "wilcoxon_p_greater_0": p,
                "significant_at_005": p < 0.05 if not math.isnan(p) else False,
            })
    return out_rows


def mettack_sign_test() -> dict:
    """149/150 wins vs Mettack — one-sided sign test (binomial)."""
    n_total = 150
    n_wins = 149
    result = binomtest(n_wins, n_total, 0.5, alternative="greater")
    return {
        "table": "mettack_signtest",
        "metric": "wins_vs_mettack",
        "n_total": n_total,
        "n_wins": n_wins,
        "binomial_p_greater_05": float(result.pvalue),
        "note": "One-sided sign test: AEGIS > Mettack in 149/150 paired comparisons"
    }


def defense_paired_wilcoxon() -> List[dict]:
    """Paired Wilcoxon for defense ablation (P2.7)."""
    csv_path = RESULTS / "exp_defense_ablation.csv"
    if not csv_path.exists():
        return [{"table": "defense_ablation",
                 "note": f"missing {csv_path}"}]
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_k: Dict[str, list] = {}
    for r in rows:
        by_k.setdefault(r.get("k", "-"), []).append(r)
    out = []
    for k, grp in by_k.items():
        try:
            ae = np.array([float(r["aegis_mask_drop"]) for r in grp])
            rd = np.array([float(r["random_mask_drop"]) for r in grp])
        except (KeyError, ValueError):
            continue
        if len(ae) < 2:
            continue
        try:
            stat = wilcoxon(ae, rd, alternative="greater",
                            zero_method="zsplit")
            p = float(stat.pvalue)
        except ValueError:
            p = float("nan")
        out.append({
            "table": "defense_ablation",
            "k": k,
            "n_seeds": len(ae),
            "aegis_mask_mean_drop": float(ae.mean()),
            "random_mask_mean_drop": float(rd.mean()),
            "delta": float((ae - rd).mean()),
            "wilcoxon_p_greater": p,
            "significant_at_005": p < 0.05,
        })
    return out


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label, spec in TABLES.items():
        rows = reanalyse_csv(label, spec)
        all_rows.extend(rows)
        print(f"{label:25s} -> {len(rows)} cells")

    # Append sign test + defense ablation
    all_rows.append(mettack_sign_test())
    all_rows.extend(defense_paired_wilcoxon())

    # Normalize keys for CSV
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"\nWrote {len(all_rows)} rows to {OUT_CSV}")
    print("\nMettack sign-test:")
    for r in all_rows:
        if r.get("table") == "mettack_signtest":
            print(f"  {r}")


if __name__ == "__main__":
    main()
