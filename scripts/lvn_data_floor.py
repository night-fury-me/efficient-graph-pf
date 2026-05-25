"""Compute trivial baselines on LVN_converted_n36000 to identify the
irreducible error floor for magnitude and angle predictions.

If the mag-RMSE plateau at 0.040 across diag A / tweak 1 / tweak 2 is a data
floor (i.e., |V_newton| ~ |V_start| on most rows), the trivial-baseline
RMSE printed here will match the trained models' RMSE. In that case,
further model tweaks are pointless — the dataset is the limit.

Run: .venv/bin/python scripts/lvn_data_floor.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_N = 1000  # ~50k bus values, plenty for stable stats
PARQUET = Path('./datasets/LVN_converted_n36000.parquet')


def decode(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    print(f'Loading {PARQUET} (sampling {SAMPLE_N} rows)...', flush=True)
    df = pd.read_parquet(PARQUET, columns=['u_start', 'u_newton', 'bus_typ'])
    df = df.iloc[:SAMPLE_N].reset_index(drop=True)
    print(f'  {len(df)} rows', flush=True)

    all_v_start = []
    all_v_newton = []
    all_vm_start = []
    all_vm_newton = []
    all_va_start = []
    all_va_newton = []

    for _, row in df.iterrows():
        u_s = decode(row.u_start)  # complex64
        u_n = decode(row.u_newton)
        all_v_start.append(u_s)
        all_v_newton.append(u_n)
        all_vm_start.append(np.abs(u_s))
        all_vm_newton.append(np.abs(u_n))
        all_va_start.append(np.angle(u_s))
        all_va_newton.append(np.angle(u_n))

    V_start = np.concatenate(all_v_start)
    V_newton = np.concatenate(all_v_newton)
    Vm_start = np.concatenate(all_vm_start)
    Vm_newton = np.concatenate(all_vm_newton)
    Va_start = np.concatenate(all_va_start)
    Va_newton = np.concatenate(all_va_newton)

    print(f'\nTotal bus samples: {len(V_start):,}\n', flush=True)

    # --- Magnitude statistics ---
    print('=== |V| statistics ===')
    print(f'  |V_newton|: mean={Vm_newton.mean():.4f}  std={Vm_newton.std():.4f}  '
          f'min={Vm_newton.min():.4f}  max={Vm_newton.max():.4f}')
    print(f'  |V_start| : mean={Vm_start.mean():.4f}  std={Vm_start.std():.4f}  '
          f'min={Vm_start.min():.4f}  max={Vm_start.max():.4f}')
    print(f'  diff:       mean={(Vm_newton-Vm_start).mean():+.4f}  '
          f'std={(Vm_newton-Vm_start).std():.4f}  '
          f'abs_max={np.abs(Vm_newton-Vm_start).max():.4f}')
    print()

    # --- Mag RMSE baselines ---
    print('=== Mag RMSE baselines (lower bound = data floor) ===')
    print(f'  baseline_flat (|V|=1.0)              : RMSE = {rmse(Vm_newton, np.ones_like(Vm_newton)):.4f}')
    print(f'  baseline_Vstart (predict |V_start|)  : RMSE = {rmse(Vm_newton, Vm_start):.4f}  <-- data floor for mag head')
    print(f'  baseline_Vmean (predict mean |V_n|)  : RMSE = {rmse(Vm_newton, np.full_like(Vm_newton, Vm_newton.mean())):.4f}')
    print()

    # --- Angle statistics ---
    print('=== arg(V) statistics (radians) ===')
    print(f'  arg(V_newton): mean={Va_newton.mean():+.4f}  std={Va_newton.std():.4f}  '
          f'min={Va_newton.min():+.4f}  max={Va_newton.max():+.4f}')
    print(f'  arg(V_start) : mean={Va_start.mean():+.4f}  std={Va_start.std():.4f}  '
          f'min={Va_start.min():+.4f}  max={Va_start.max():+.4f}')
    diff_ang_rad = Va_newton - Va_start
    # Wrap to [-pi, pi]
    diff_ang_rad = ((diff_ang_rad + np.pi) % (2 * np.pi)) - np.pi
    print(f'  diff (wrapped): mean={diff_ang_rad.mean():+.4f}  std={diff_ang_rad.std():.4f}  '
          f'abs_max={np.abs(diff_ang_rad).max():.4f}  ({np.degrees(np.abs(diff_ang_rad).max()):.2f}deg)')
    print()

    # --- Ang RMSE baselines ---
    print('=== Ang RMSE baselines (in degrees) ===')
    print(f'  baseline_flat (arg=0)                 : RMSE = {np.degrees(rmse(Va_newton, np.zeros_like(Va_newton))):.4f}deg')
    print(f'  baseline_Vstart (predict arg(V_start)): RMSE = {np.degrees(rmse(Va_newton, Va_start)):.4f}deg  <-- data floor for ang head')
    print()

    # --- Combined V RMSE (matches model's mse_components) ---
    # The training mse is on complex V; the "mag" and "ang" RMSEs reported in
    # the log are derived components. Replicate them.
    # Looking at mse_components: it's likely real/imag separated, summed.
    print('=== Complex-V RMSE baselines (matches train log "rmse" column) ===')
    # rmse over complex separated as real and imag (matches L2 norm)
    re_diff = V_newton.real - V_start.real
    im_diff = V_newton.imag - V_start.imag
    rmse_complex = float(np.sqrt(np.mean(re_diff**2 + im_diff**2) / 2))
    print(f'  predict V_start : sqrt(mean( (re_diff^2 + im_diff^2)/2 )) = {rmse_complex:.4f}')
    print(f'  predict V=1+0j  : ',
          float(np.sqrt(np.mean(((V_newton.real - 1.0)**2 + V_newton.imag**2) / 2))))
    print()

    # --- Interpretation ---
    mag_floor = rmse(Vm_newton, Vm_start)
    print('=== INTERPRETATION ===')
    print(f'  Trained models converge to: val mag RMSE = 0.0403')
    print(f'  Trivial V_start baseline  : val mag RMSE = {mag_floor:.4f}')
    if abs(mag_floor - 0.0403) < 0.002:
        print(f'  ==> MATCH. The 0.0403 mag floor is the data floor.')
        print(f'      No model tweak can improve magnitude further on this dataset.')
        print(f'      The "magnitude error" is just |V_newton - V_start|, which the')
        print(f'      model already matches by predicting near-V_start.')
    elif mag_floor > 0.0403:
        print(f'  ==> Model BEATS the V_start baseline. Tweaks could help further.')
    else:
        print(f'  ==> Model is WORSE than V_start baseline. There may be a head bug.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
