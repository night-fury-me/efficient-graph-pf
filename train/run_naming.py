from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _short_model_tag(model_name: str) -> str:
    s = model_name
    s = re.sub(r"^GNSMsg_?", "", s)
    s = s.replace("SelfAttention", "Attn")
    s = s.replace("SelfAttn", "Attn")
    s = s.replace("EdgeSelfAttn", "EdgeAttn")
    s = s.replace("Edge", "Edge")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_").lower()


def _short_dataset_tag(parquet_paths: Iterable[str]) -> str:
    paths = list(parquet_paths)
    n = len(paths)
    if n == 0:
        return "ds_n0"

    # Compute a stable short hash over basenames (order-independent).
    bases = sorted(Path(p).stem for p in paths)
    digest = hashlib.sha1("|".join(bases).encode("utf-8")).hexdigest()[:6]

    # Heuristic: names like HVN_15000_NR_plain_...
    t0 = bases[0].split("_")
    if len(t0) >= 3 and t0[1].isdigit():
        prefix = t0[0].lower()
        scenario = t0[2].lower()
        return f"{prefix}_{scenario}_n{n}_{digest}"

    return f"ds_n{n}_{digest}"


def make_run_slug(
    *,
    parquet_paths: Iterable[str],
    model_name: str,
    K: int,
    d: int,
    d_hi: int,
    pinn: bool,
    block_diag: bool,
    per_unit: bool,
    split_mode: str,
) -> str:
    ds = _short_dataset_tag(parquet_paths)
    model = _short_model_tag(model_name)
    flags: list[str] = []
    if pinn:
        flags.append("pinn")
    if block_diag:
        flags.append("blk")
    if per_unit:
        flags.append("pu")
    if split_mode == "equal3":
        flags.append("eq3")

    core = f"{ds}_{model}_k{K}_d{d}_h{d_hi}"
    if flags:
        core = core + "_" + "_".join(flags)
    return core


def make_run_id(*, run_slug: str) -> str:
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    h = hashlib.sha1(run_slug.encode("utf-8")).hexdigest()[:4]
    return f"{ts}_{h}"


def safe_param_dict(cfg) -> dict[str, str | int | float | bool]:
    """Flatten dataclass-like configs into MLflow-safe param values."""
    raw = asdict(cfg)
    out: dict[str, str | int | float | bool] = {}
    for k, v in raw.items():
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ",".join(str(x) for x in v)
        elif v is None:
            out[k] = "null"
        else:
            out[k] = str(v)
    return out
