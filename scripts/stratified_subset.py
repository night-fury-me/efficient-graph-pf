#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("stratified_subset")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a stratified subset from parquet files. "
            "Default strata: gridtype (voltage regime) and bus_number (grid size)."
        )
    )
    p.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="Input parquet path(s) to combine.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output parquet path for the stratified subset.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Total number of samples to select.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    p.add_argument(
        "--strata",
        nargs="+",
        default=["gridtype", "bus_number"],
        help="Column names to use for stratification.",
    )
    p.add_argument(
        "--min-per-stratum",
        type=int,
        default=1,
        help="Minimum samples per non-empty stratum (0 to disable).",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a JSON summary of counts.",
    )
    p.add_argument(
        "--post-cleaning",
        action="store_true",
        help=(
            "Apply dataset cleaning (remove diverged rows and per-unit outliers) before sampling. "
            "This mirrors the cleaning in data_loading/dataset.py."
        ),
    )
    p.add_argument(
        "--outlier-k",
        type=float,
        default=1.5,
        help="IQR multiplier for per-unit outlier removal (default: 1.5).",
    )
    return p.parse_args(argv)


def _safe_list(val) -> list:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            import ast

            return ast.literal_eval(val)
    return list(val)


def _to_complex_array(val) -> np.ndarray:
    lst = _safe_list(val)
    obj = np.array(lst, dtype=object).reshape(-1)

    def to_c(e):
        if isinstance(e, complex) or np.issubdtype(type(e), np.complexfloating):
            return e
        if isinstance(e, (list, tuple)) and len(e) == 2:
            return complex(e[0], e[1])
        if isinstance(e, dict):
            if "real" in e and "imag" in e:
                return complex(e["real"], e["imag"])
            if "re" in e and "im" in e:
                return complex(e["re"], e["im"])
        if isinstance(e, str):
            s = e.strip().replace("i", "j")
            try:
                return complex(s)
            except Exception:
                s2 = s.strip("()[]")
                parts = [p.strip() for p in s2.split(",")]
                if len(parts) == 2:
                    return complex(float(parts[0]), float(parts[1]))
                raise
        if isinstance(e, (int, float, np.integer, np.floating)):
            return complex(e, 0.0)
        return complex(float(e), 0.0)

    return np.array([to_c(e) for e in obj], dtype=np.complex64)


def _is_zero_complex_list(val) -> bool:
    arr = _to_complex_array(val)
    return arr.size > 0 and np.all(arr == 0)


def _iqr_bounds(data: Sequence[float], k: float) -> Optional[tuple[float, float]]:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def _apply_dataset_cleaning(df: pd.DataFrame, *, outlier_k: float) -> pd.DataFrame:
    df = df.copy()

    # remove diverged rows where u_newton is all zeros
    if "u_newton" in df.columns:
        keep = ~df["u_newton"].map(_is_zero_complex_list)
        if not keep.all():
            df = df[keep].reset_index(drop=True)
            log.info("Removed %s diverged rows -> %s", int((~keep).sum()), df.shape)

    # remove per-unit outliers in u_newton if U_base exists
    if "u_newton" in df.columns and "U_base" in df.columns:
        u_bases = df["U_base"].astype(float).to_numpy()
        all_real_pu: list[float] = []
        all_imag_pu: list[float] = []

        for cell, u_base in zip(df["u_newton"], u_bases):
            if not np.isfinite(u_base) or u_base == 0:
                continue
            try:
                u = _to_complex_array(cell)
            except Exception:
                continue
            finite_mask = np.isfinite(u.real) & np.isfinite(u.imag)
            u = u[finite_mask]
            if u.size == 0:
                continue
            all_real_pu.extend((u.real / u_base).tolist())
            all_imag_pu.extend((u.imag / u_base).tolist())

        b_real = _iqr_bounds(all_real_pu, outlier_k)
        b_imag = _iqr_bounds(all_imag_pu, outlier_k)

        if b_real is not None or b_imag is not None:
            lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
            li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)

            mask_row_outlier = np.zeros(len(df), dtype=bool)

            for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
                if not np.isfinite(u_base) or u_base == 0:
                    continue
                try:
                    u = _to_complex_array(cell)
                except Exception:
                    continue
                if u.size == 0:
                    continue
                r_pu = np.asarray(u.real / u_base, dtype=float)
                i_pu = np.asarray(u.imag / u_base, dtype=float)
                r_pu = r_pu[np.isfinite(r_pu)]
                i_pu = i_pu[np.isfinite(i_pu)]
                bad_r = r_pu.size > 0 and (np.any(r_pu < lr) or np.any(r_pu > ur))
                bad_i = i_pu.size > 0 and (np.any(i_pu < li) or np.any(i_pu > ui))
                if bad_r or bad_i:
                    mask_row_outlier[idx] = True

            n_bad = int(mask_row_outlier.sum())
            if n_bad > 0:
                df = df[~mask_row_outlier].reset_index(drop=True)
                log.info("Removed %s per-unit outlier rows -> %s", n_bad, df.shape)

    return df


def _allocate_counts(
    counts: pd.Series,
    *,
    n_samples: int,
    min_per_stratum: int,
) -> pd.Series:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    counts = counts.sort_index()
    n_strata = int(counts.shape[0])
    if n_strata == 0:
        raise ValueError("No strata found. Check strata columns.")

    if min_per_stratum < 0:
        raise ValueError("min_per_stratum must be >= 0")

    if min_per_stratum > 0:
        min_total = n_strata * min_per_stratum
        if n_samples < min_total:
            raise ValueError(
                f"n_samples={n_samples} is smaller than n_strata*min_per_stratum={min_total}."
            )
        base = pd.Series(min_per_stratum, index=counts.index, dtype=int)
        remaining = n_samples - min_total
    else:
        base = pd.Series(0, index=counts.index, dtype=int)
        remaining = n_samples

    if remaining <= 0:
        return base

    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Total count across strata is 0")

    raw = counts / total * remaining
    floor = np.floor(raw).astype(int)
    remainder = int(remaining - int(floor.sum()))
    frac = (raw - floor).sort_values(ascending=False)

    alloc = floor.copy()
    if remainder > 0:
        take = frac.index[:remainder]
        alloc.loc[take] += 1

    return base + alloc


def _sample_by_strata(
    df: pd.DataFrame,
    strata_cols: list[str],
    target_counts: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sample_indices: list[int] = []

    grouped = df.groupby(strata_cols, sort=False, dropna=False)
    for key, idxs in grouped.indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        count = int(target_counts.loc[key])
        if count <= 0:
            continue
        idxs_arr = np.asarray(list(idxs), dtype=int)
        if idxs_arr.size < count:
            raise ValueError(
                f"Stratum {key} has only {idxs_arr.size} rows, but {count} requested."
            )
        rng.shuffle(idxs_arr)
        sample_indices.extend(idxs_arr[:count].tolist())

    return df.loc[sample_indices].reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parquet_paths = [Path(p) for p in args.parquet]
    for p in parquet_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    strata_cols = list(args.strata)
    n_samples = int(args.n_samples)
    min_per_stratum = int(args.min_per_stratum)

    log.info("Loading %d parquet files", len(parquet_paths))
    frames = [pd.read_parquet(p, engine="pyarrow") for p in parquet_paths]
    df = pd.concat(frames, axis=0, ignore_index=True)

    if args.post_cleaning:
        df = _apply_dataset_cleaning(df, outlier_k=float(args.outlier_k))

    missing = [c for c in strata_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing strata columns: {missing}")

    counts = df.groupby(strata_cols, dropna=False).size()
    target = _allocate_counts(counts, n_samples=n_samples, min_per_stratum=min_per_stratum)

    rng = np.random.default_rng(int(args.seed))
    subset = _sample_by_strata(df, strata_cols, target, rng)

    if len(subset) != n_samples:
        raise SystemExit(f"Unexpected sample count: {len(subset)} vs {n_samples}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out_path, index=False)

    def _stringify_keys(d: dict) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                key = "|".join(str(x) for x in k)
            else:
                key = str(k)
            out[key] = int(v)
        return out

    summary = {
        "n_total": int(len(df)),
        "n_subset": int(len(subset)),
        "seed": int(args.seed),
        "strata_cols": strata_cols,
        "input_parquets": [str(p) for p in parquet_paths],
        "counts_by_stratum": _stringify_keys(counts.to_dict()),
        "subset_counts_by_stratum": _stringify_keys(
            subset.groupby(strata_cols, dropna=False).size().to_dict()
        ),
    }

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Wrote subset to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
