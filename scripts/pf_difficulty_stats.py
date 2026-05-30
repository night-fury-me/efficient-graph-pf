#!/usr/bin/env python3
"""Compute dataset-level difficulty / stress indicators for stored PF solutions.

This repo's parquet datasets store, per case:
  - complex bus voltages from a Newton(-Raphson) solve: `u_newton`
  - bus type codes: `bus_typ` (1=slack, 2=PV, 3=PQ)
  - network admittance matrix (flattened): `Y_matrix`
  - base values: `U_base`, `S_base`

However, the parquet schema does *not* include continuation power flow / PV curves,
loading margin, or NR iteration counts. This script adds *concrete, reproducible*
proxy metrics that can be computed from what is already stored:

  1) Low-voltage indicator: v_min_pu = min_i |V_i| (per-unit)
  2) NR power-flow Jacobian conditioning at the stored operating point:
      sigma_min(J) and cond_2(J)

The Jacobian here is the standard power-flow Jacobian for bus injection
mismatches w.r.t. state variables (theta for PV/PQ, V for PQ) with the usual
row/column elimination for slack/PV constraints.

Outputs:
  - Per-case CSV (optional)
  - Summary JSON with quantiles + fractions over thresholds

Example:
  uv run python scripts/pf_difficulty_stats.py \
    --parquet datasets/HVN_stratified_500_postclean.parquet \
    --out-json results/pf_difficulty/HVN500_stats.json \
    --out-csv  results/pf_difficulty/HVN500_cases.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Local import: decode helper already exists in-repo.
from data_loading.npy_decode import npy_bytes_to_ndarray

log = logging.getLogger("pf_difficulty")


SLACK = 1
PV = 2
PQ = 3


def _decode_complex_1d(b: bytes) -> np.ndarray:
    a = npy_bytes_to_ndarray(b)
    # Stored as complex dtype already in most datasets.
    return np.asarray(a, dtype=np.complex128).reshape(-1)


def _is_all_zero_complex(b: bytes) -> bool:
    u = _decode_complex_1d(b)
    return bool(u.size > 0 and np.all(u == 0))


def _ybus_pu_from_row(y_flat_b: bytes, *, s_base: float, u_base: float) -> np.ndarray:
    yflat = npy_bytes_to_ndarray(y_flat_b)
    yflat = np.asarray(yflat, dtype=np.complex128).reshape(-1)
    n = int(round(np.sqrt(yflat.size)))
    if n * n != yflat.size:
        raise ValueError(f"Y_matrix length {yflat.size} is not a perfect square")

    Y = yflat.reshape(n, n)
    # Per-unit scaling consistent with data_loading/dataset.py
    y_base = s_base / (u_base**2)
    return Y / y_base


def _vj_theta_from_row(u_newton_b: bytes, *, u_base: float) -> tuple[np.ndarray, np.ndarray]:
    vpu_c = _decode_complex_1d(u_newton_b) / float(u_base)
    v = np.abs(vpu_c)
    th = np.angle(vpu_c)
    return v, th


def _calc_pq(Y: np.ndarray, v: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute bus injections P,Q (per-unit) implied by (Y, v, th)."""
    G = Y.real
    B = Y.imag

    # Angle differences: θ_i - θ_j
    dth = th[:, None] - th[None, :]
    c = np.cos(dth)
    s = np.sin(dth)

    # P_i = sum_j V_i V_j (G_ij cos + B_ij sin)
    # Q_i = sum_j V_i V_j (G_ij sin - B_ij cos)
    vv = v[:, None] * v[None, :]
    P = np.sum(vv * (G * c + B * s), axis=1)
    Q = np.sum(vv * (G * s - B * c), axis=1)
    return P, Q


def _pf_jacobian(Y: np.ndarray, v: np.ndarray, th: np.ndarray, bus_type: np.ndarray) -> np.ndarray:
    """Build reduced NR power-flow Jacobian at (v, th).

    Variables:
      - θ for all PV+PQ buses (slack removed)
      - V for PQ buses only

    Equations:
      - P mismatch for all PV+PQ buses (slack removed)
      - Q mismatch for PQ buses only

        Notes:
            - This is the *standard* PF Jacobian for the bus-injection model, not the
                network admittance matrix.

        Returns:
      Square Jacobian matrix with shape (n_var, n_var).
    """

    bus_type = np.asarray(bus_type, dtype=int).reshape(-1)
    n = int(bus_type.size)
    if v.size != n or th.size != n:
        raise ValueError("v/th/bus_type size mismatch")

    # Index sets
    idx_theta = np.flatnonzero(bus_type != SLACK)
    idx_v = np.flatnonzero(bus_type == PQ)

    # No PQ buses => no V variables; still valid (Jacobian is dP/dθ).
    # If there are only slack buses (degenerate), reject.
    if idx_theta.size == 0:
        raise ValueError("No non-slack buses; cannot form Jacobian")

    # Precompute P,Q (per-unit)
    P, Q = _calc_pq(Y, v, th)

    G = Y.real
    B = Y.imag
    dth = th[:, None] - th[None, :]
    c = np.cos(dth)
    s = np.sin(dth)

    vv = v[:, None] * v[None, :]

    # Full (unreduced) partials (n x n)
    # Off-diagonals i!=j use closed forms
    dP_dth = vv * (G * s - B * c)
    dQ_dth = -vv * (G * c + B * s)
    dP_dV = v[:, None] * (G * c + B * s)
    dQ_dV = v[:, None] * (G * s - B * c)

    # Diagonal corrections
    # Set diagonals explicitly to standard NR forms.
    for i in range(n):
        dP_dth[i, i] = -Q[i] - B[i, i] * (v[i] ** 2)
        dQ_dth[i, i] = P[i] - G[i, i] * (v[i] ** 2)
        if v[i] == 0:
            # Avoid division by zero; treat as ill-conditioned.
            dP_dV[i, i] = np.inf
            dQ_dV[i, i] = np.inf
        else:
            dP_dV[i, i] = P[i] / v[i] + G[i, i] * v[i]
            dQ_dV[i, i] = Q[i] / v[i] - B[i, i] * v[i]

    # Reduced blocks
    rows_P = idx_theta
    rows_Q = idx_v

    J11 = dP_dth[np.ix_(rows_P, idx_theta)]
    J12 = dP_dV[np.ix_(rows_P, idx_v)] if idx_v.size else np.zeros((rows_P.size, 0), dtype=float)
    J21 = dQ_dth[np.ix_(rows_Q, idx_theta)] if rows_Q.size else np.zeros((0, idx_theta.size), dtype=float)
    J22 = dQ_dV[np.ix_(rows_Q, idx_v)] if (rows_Q.size and idx_v.size) else np.zeros((rows_Q.size, idx_v.size), dtype=float)

    top = np.concatenate([J11, J12], axis=1)
    bot = np.concatenate([J21, J22], axis=1)
    J = np.concatenate([top, bot], axis=0)

    if J.shape[0] != J.shape[1]:
        raise ValueError(f"Jacobian is not square after reduction: {J.shape}")

    return J


def _finite_stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}

    qs = [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]
    qv = np.quantile(x, qs)
    return {
        "n": int(x.size),
        "min": float(qv[0]),
        "p01": float(qv[1]),
        "p05": float(qv[2]),
        "p10": float(qv[3]),
        "p50": float(qv[4]),
        "p90": float(qv[5]),
        "p95": float(qv[6]),
        "p99": float(qv[7]),
        "max": float(qv[8]),
        "mean": float(np.mean(x)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute PF difficulty/stress proxies from parquet datasets")
    p.add_argument("--parquet", nargs="+", required=True, help="Input parquet path(s).")

    p.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path for summary stats.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path with per-case metrics.",
    )

    p.add_argument("--max-rows", type=int, default=None, help="Optionally limit number of rows for a quick run.")

    p.add_argument("--vmin-thr", type=float, default=0.90, help="Threshold for v_min_pu fraction reporting.")
    p.add_argument(
        "--cond-thr",
        type=float,
        default=1e4,
        help="Threshold for cond_2(J) fraction reporting.",
    )
    p.add_argument(
        "--sigma-min-thr",
        type=float,
        default=None,
        help="Optional threshold for sigma_min(J) fraction reporting.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parquet_paths = [Path(x) for x in args.parquet]
    for pth in parquet_paths:
        if not pth.exists():
            raise FileNotFoundError(pth)

    log.info("Loading %d parquet files", len(parquet_paths))
    frames = [pd.read_parquet(p, engine="pyarrow") for p in parquet_paths]
    df = pd.concat(frames, axis=0, ignore_index=True)

    if args.max_rows is not None:
        df = df.iloc[: int(args.max_rows)].reset_index(drop=True)

    required = {"bus_typ", "Y_matrix", "u_newton", "U_base", "S_base"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Identify diverged rows (convention used in data_loading/dataset.py)
    diverged = df["u_newton"].map(_is_all_zero_complex).to_numpy()
    n_div = int(diverged.sum())
    if n_div:
        log.info("Detected %d diverged rows with all-zero u_newton (will exclude from metrics)", n_div)

    df_ok = df.loc[~diverged].reset_index(drop=True)

    vmin_pu: list[float] = []
    sigma_min: list[float] = []
    cond2: list[float] = []

    failures: list[str] = []

    for i, r in df_ok.iterrows():
        try:
            bus_type = npy_bytes_to_ndarray(r.bus_typ).astype(int).reshape(-1)
            u_base = float(r.U_base)
            s_base = float(r.S_base)

            v, th = _vj_theta_from_row(r.u_newton, u_base=u_base)
            vmin_pu.append(float(np.min(v)))

            Ypu = _ybus_pu_from_row(r.Y_matrix, s_base=s_base, u_base=u_base)
            J = _pf_jacobian(Ypu, v, th, bus_type)

            svals = np.linalg.svd(J, compute_uv=False)
            smax = float(svals[0])
            smin = float(svals[-1])
            sigma_min.append(smin)
            cond2.append(np.inf if smin == 0.0 else (smax / smin))
        except Exception as e:
            failures.append(f"row={i}: {type(e).__name__}: {e}")

    vmin_pu_a = np.asarray(vmin_pu, dtype=float)
    sigma_min_a = np.asarray(sigma_min, dtype=float)
    cond2_a = np.asarray(cond2, dtype=float)

    summary: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_diverged_all_zero_u_newton": int(n_div),
        "n_used": int(len(df_ok)),
        "vmin_pu": _finite_stats(vmin_pu_a),
        "sigma_min_J": _finite_stats(sigma_min_a),
        "cond2_J": _finite_stats(cond2_a),
        "thresholds": {
            "vmin_thr": float(args.vmin_thr),
            "cond_thr": float(args.cond_thr),
            "sigma_min_thr": (None if args.sigma_min_thr is None else float(args.sigma_min_thr)),
        },
        "fractions": {
            "vmin_below_thr": float(np.mean(vmin_pu_a < float(args.vmin_thr))) if vmin_pu_a.size else None,
            "cond_above_thr": float(np.mean(cond2_a > float(args.cond_thr))) if cond2_a.size else None,
            "sigma_min_below_thr": (
                None
                if args.sigma_min_thr is None or sigma_min_a.size == 0
                else float(np.mean(sigma_min_a < float(args.sigma_min_thr)))
            ),
        },
        "n_failures": int(len(failures)),
        "failures_head": failures[:10],
        "input_parquets": [str(p) for p in parquet_paths],
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Wrote summary JSON -> %s", out_json)
    else:
        # Print a compact summary to stdout.
        print(json.dumps(summary, indent=2))

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cases = pd.DataFrame(
            {
                "vmin_pu": vmin_pu_a,
                "sigma_min_J": sigma_min_a,
                "cond2_J": cond2_a,
            }
        )
        cases.to_csv(out_csv, index=False)
        log.info("Wrote per-case CSV -> %s", out_csv)

    if failures:
        log.warning("Encountered %d failures while computing metrics (showing first 3):", len(failures))
        for line in failures[:3]:
            log.warning("  %s", line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
