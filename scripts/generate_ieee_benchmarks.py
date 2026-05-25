"""Generate IEEE benchmark datasets for IEM power flow experiments.

Creates multiple operating points for each IEEE test case by varying
loads ±30% and solving Newton-Raphson. Outputs parquet files matching
the HVN/MVN schema (bus_typ, Y_matrix, u_start, u_newton, S_start, etc.)
so they can be loaded directly by ChanghunDataset.

Cases: IEEE 14, 30, 57, 118 (standard PF-ML benchmarks).
Samples: 2000 per case (enough for train/val/test).

Usage:
    .venv/bin/python scripts/generate_ieee_benchmarks.py
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pandapower as pp
import pandapower.networks as pn


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def generate_case(case_name: str, net_fn, n_samples: int = 2000, load_var: float = 0.3):
    """Generate multiple operating points for one IEEE case."""
    print(f"\n=== {case_name} ===", flush=True)

    net = net_fn()
    n_bus = len(net.bus)
    print(f"  {n_bus} buses, {len(net.line)} lines, {len(net.trafo)} trafos", flush=True)

    # Get base load values
    base_p_load = net.load.p_mw.values.copy()
    base_q_load = net.load.q_mvar.values.copy()
    base_p_gen = net.gen.p_mw.values.copy() if len(net.gen) > 0 else np.array([])

    # Solve base case to get Y_bus
    pp.runpp(net, init="flat", numba=True, enforce_q_lims=False)
    Y_bus = net._ppc["internal"]["Ybus"].toarray().astype(np.complex64)

    # Bus type mapping: pandapower → our convention (0=PQ, 1=slack, 2=PV)
    # pandapower: 1=PQ, 2=PV, 3=ref/slack (in the ppc)
    bus_typ_ppc = net._ppc["bus"][:, 1].astype(int)  # BUS_TYPE column
    bus_typ = np.zeros(n_bus, dtype=np.int32)
    bus_typ[bus_typ_ppc == 1] = 0  # PQ
    bus_typ[bus_typ_ppc == 2] = 2  # PV
    bus_typ[bus_typ_ppc == 3] = 1  # slack

    S_base = float(net._ppc["baseMVA"]) * 1e6  # in VA
    U_base = net.bus.vn_kv.values[0] * 1000.0   # in V (use first bus)

    rows = []
    n_success = 0
    rng = np.random.RandomState(42)

    for i in range(n_samples * 2):  # oversample to handle NR failures
        if n_success >= n_samples:
            break

        # Vary loads uniformly ±load_var
        scale_p = 1.0 + rng.uniform(-load_var, load_var, size=len(base_p_load))
        scale_q = 1.0 + rng.uniform(-load_var, load_var, size=len(base_q_load))
        net.load.p_mw = base_p_load * scale_p
        net.load.q_mvar = base_q_load * scale_q

        # Vary generation ±10%
        if len(base_p_gen) > 0:
            scale_g = 1.0 + rng.uniform(-0.1, 0.1, size=len(base_p_gen))
            net.gen.p_mw = base_p_gen * scale_g

        try:
            pp.runpp(net, init="flat", numba=True, enforce_q_lims=False,
                     max_iteration=30, tolerance_mva=1e-6)
        except pp.powerflow.LoadflowNotConverged:
            continue

        if not net.converged:
            continue

        # Extract results
        vm = net.res_bus.vm_pu.values.astype(np.float64)
        va = np.radians(net.res_bus.va_degree.values).astype(np.float64)
        u_newton = (vm * np.exp(1j * va)).astype(np.complex64)

        # Flat start
        u_start = np.ones(n_bus, dtype=np.complex64)
        # PV buses: set voltage magnitude to specified value
        for _, gen in net.gen.iterrows():
            bus_idx = int(gen.bus)
            u_start[bus_idx] = complex(float(gen.vm_pu), 0.0)

        # S injection (per-unit)
        p_bus = net.res_bus.p_mw.values / (S_base / 1e6)
        q_bus = net.res_bus.q_mvar.values / (S_base / 1e6)
        S_start = (p_bus + 1j * q_bus).astype(np.complex64)

        # Lines connected (upper triangular canonical pairs)
        n_pairs = n_bus * (n_bus - 1) // 2
        lines_connected = np.zeros(n_pairs, dtype=np.bool_)
        Y_lines = np.zeros(n_pairs, dtype=np.complex64)
        Y_C_lines = np.zeros(n_pairs, dtype=np.float64)

        for ii in range(n_bus):
            for jj in range(ii + 1, n_bus):
                if abs(Y_bus[ii, jj]) > 1e-12:
                    k = ii * n_bus - ii * (ii + 1) // 2 + (jj - ii - 1)
                    lines_connected[k] = True
                    Y_lines[k] = -Y_bus[ii, jj]  # off-diagonal is -y_ij
                    Y_C_lines[k] = 0.0  # shunt absorbed in Y_bus diagonal

        rows.append({
            "bus_number": n_bus,
            "gridtype": f"IEEE_{case_name}",
            "U_base": U_base,
            "S_base": S_base,
            "bus_typ": npy_bytes(bus_typ),
            "Y_Lines": npy_bytes(Y_lines),
            "Y_C_Lines": npy_bytes(Y_C_lines),
            "Lines_connected": npy_bytes(lines_connected),
            "Y_matrix": npy_bytes(Y_bus),
            "u_start": npy_bytes(u_start),
            "u_newton": npy_bytes(u_newton),
            "S_start": npy_bytes(S_start),
            "S_newton": npy_bytes(S_start),  # placeholder
            "I_newton": npy_bytes(np.zeros_like(S_start)),
        })
        n_success += 1

        if n_success % 500 == 0:
            print(f"  {n_success}/{n_samples} samples generated", flush=True)

    print(f"  Generated {n_success} samples (from {i+1} attempts)", flush=True)
    return rows


def main():
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)

    cases = [
        ("case14", pn.case14, 2000),
        ("case30", pn.case_ieee30, 2000),
        ("case57", pn.case57, 2000),
        ("case118", pn.case118, 2000),
    ]

    for case_name, net_fn, n_samples in cases:
        rows = generate_case(case_name, net_fn, n_samples=n_samples)
        if not rows:
            print(f"  SKIPPED — no converged samples", flush=True)
            continue

        df = pd.DataFrame(rows)
        out_path = out_dir / f"IEEE_{case_name}_{len(rows)}.parquet"
        df.to_parquet(out_path, compression="snappy")
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)", flush=True)

    print("\n=== ALL IEEE BENCHMARKS GENERATED ===", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
