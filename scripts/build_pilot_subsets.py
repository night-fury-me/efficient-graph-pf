"""Build stratified subsets of HVN and MVN for the HyperDEQ-PF pilot.

Subset spec:
- Source: HVN_15000_NR_plain_4_to_32_buses.parquet and MVN_30000_NR_plain_4_to_32_buses.parquet
- Stratified by bus_number (29 strata: 4..32)
- ~50 samples per stratum -> ~1500 per dataset
- Deterministic via seed=42 (matches existing run convention)

Output:
- datasets/HVN_stratified_1500.parquet
- datasets/MVN_stratified_1500.parquet

Run: .venv/bin/python scripts/build_pilot_subsets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SAMPLES_PER_STRATUM = 50


def stratify(src: Path, out: Path, samples_per_stratum: int = SAMPLES_PER_STRATUM) -> None:
    print(f"Loading {src} ...", flush=True)
    df = pd.read_parquet(src)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    rng = np.random.default_rng(SEED)
    picked: list[int] = []
    strata = sorted(df.bus_number.unique())
    for n in strata:
        idx = np.where(df.bus_number.values == n)[0]
        if len(idx) <= samples_per_stratum:
            picked.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=samples_per_stratum, replace=False)
            picked.extend(chosen.tolist())

    picked.sort()
    sub = df.iloc[picked].reset_index(drop=True)
    print(f"  stratified subset: {len(sub)} rows over {len(strata)} bus-counts (4..{strata[-1]})")
    print(f"  per-stratum distribution: {sub.bus_number.value_counts().sort_index().tolist()[:5]} ... {sub.bus_number.value_counts().sort_index().tolist()[-5:]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out, compression="snappy")
    size_mb = out.stat().st_size / 1e6
    print(f"  wrote {out}  ({size_mb:.1f} MB)\n")


def main() -> int:
    root = Path(".")
    sources = [
        (root / "datasets/HVN_15000_NR_plain_4_to_32_buses.parquet",
         root / "datasets/HVN_stratified_1500.parquet"),
        (root / "datasets/MVN_30000_NR_plain_4_to_32_buses.parquet",
         root / "datasets/MVN_stratified_1500.parquet"),
    ]
    for src, out in sources:
        if not src.exists():
            print(f"ERROR: source not found: {src}", file=sys.stderr)
            return 1
        stratify(src, out)

    print("Done. Pilot subsets ready:")
    print("  - datasets/HVN_stratified_1500.parquet")
    print("  - datasets/MVN_stratified_1500.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
