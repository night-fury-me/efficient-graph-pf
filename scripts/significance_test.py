#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from rich.progress import track

# Ensure repo root on sys.path (avoid scripts/ shadowing train/ package)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.config_loader import as_path_list, deep_update, env_expand_paths, get, load_yaml_config
from train.data import build_dataloaders
from train.modeling import create_model
from train.peft_utils import apply_lora_to_linear_modules


log = logging.getLogger("significance_test")


def _ensure_mapping(x: Any, *, what: str) -> dict[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"Expected {what} to be a mapping/dict, got: {type(x).__name__}")
    return dict(x)


def _load_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a state_dict or {{'state_dict': ...}} mapping, got {type(payload).__name__}")
    return payload  # type: ignore[return-value]


def _looks_like_lora_state_dict(sd: Mapping[str, Any]) -> bool:
    return any(".lora_A" in str(k) or ".lora_B" in str(k) for k in sd.keys())


def _detect_lora_rank_from_state_dict(sd: Mapping[str, Any]) -> int | None:
    ranks: set[int] = set()
    for k, v in sd.items():
        if ".lora_A" not in str(k):
            continue
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        ranks.add(int(v.shape[0]))
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Detected multiple LoRA ranks in checkpoint: {sorted(ranks)}")
    return next(iter(ranks))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run paired Wilcoxon signed-rank tests against Full FT using per-sample RMSE, "
            "and save per-sample errors and summary CSVs."
        )
    )

    cfg = p.add_mutually_exclusive_group(required=True)
    cfg.add_argument(
        "--config",
        default=None,
        help="Single YAML config file (already merged).",
    )
    cfg.add_argument(
        "--base",
        "--base-config",
        dest="base_config",
        default=None,
        help="Base YAML config (e.g., configs/default.yaml). Use with --scenario-* for overlays.",
    )

    p.add_argument(
        "--scenario",
        "--scenario-config",
        dest="scenario_config",
        default=None,
        help="Optional scenario overlay YAML to use for all methods (fallback).",
    )

    p.add_argument("--scenario-full", default=None, help="Scenario YAML for Full FT")
    p.add_argument("--scenario-head", default=None, help="Scenario YAML for Head-only")
    p.add_argument("--scenario-lora", default=None, help="Scenario YAML for LoRA-only")
    p.add_argument("--scenario-lora-head", default=None, help="Scenario YAML for LoRA+Head")

    p.add_argument("--full-ckpt", required=True, help="Full FT checkpoint path")
    p.add_argument("--head-ckpt", default=None, help="Head-only checkpoint path")
    p.add_argument("--lora-ckpt", default=None, help="LoRA-only checkpoint path")
    p.add_argument("--lora-head-ckpt", default=None, help="LoRA+Head checkpoint path")

    p.add_argument(
        "--parquet",
        nargs="+",
        default=None,
        help=(
            "Parquet path(s) for evaluation. Overrides data.parquet_paths in config. "
            "Example: --parquet ./datasets/MVN_30000_...parquet ./datasets/MVN_30010_...parquet"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override split seed (must be identical across methods).",
    )
    p.add_argument("--device", default=None, help="Device override (e.g., cuda, cuda:0, cpu).")

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (defaults to min(config.train.batch_size, 32)).",
    )

    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/significance_tests/<timestamp>).",
    )

    # LoRA overrides (used when a checkpoint is detected as LoRA)
    p.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank r")
    p.add_argument("--lora-alpha", type=int, default=None, help="Override LoRA alpha")
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="Override LoRA dropout",
    )
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Override LoRA target module names (e.g., q k v out)",
    )

    return p


def _angle_wrap_delta(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _per_sample_rmse(
    *,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
) -> np.ndarray:
    model.eval()
    errs: list[float] = []

    try:
        total = len(test_loader)
    except Exception:
        total = None

    with torch.no_grad():
        for batch in track(test_loader, total=total, description="eval", transient=True):
            n_nodes_per_graph = batch["sizes"].to(device) if block_diag else None

            bus_type = batch["bus_type"].to(device)
            Line = batch["Lines_connected"].to(device)
            Y_raw = batch.get("Ybus", None)
            Y = Y_raw.to(device, non_blocking=True) if isinstance(Y_raw, torch.Tensor) else None
            Ys = batch["Y_Lines"].to(device)
            Yc = batch["Y_C_Lines"].to(device)

            Sstart = batch["S_start"].to(device)
            Vstart = batch["V_start"].to(device)
            Vref = batch["V_newton"].to(device)

            if pinn:
                Vpred, _ = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
            else:
                Vpred = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)

            if block_diag:
                sizes = batch["sizes"].to(device)
                offsets = batch["offsets"].to(device)
                Vpred0 = Vpred[0]
                Vref0 = Vref[0]

                for i, n in enumerate(sizes.tolist()):
                    start = int(offsets[i].item())
                    end = start + int(n)
                    dmag = Vpred0[start:end, 0] - Vref0[start:end, 0]
                    dang = _angle_wrap_delta(Vpred0[start:end, 1] - Vref0[start:end, 1])
                    mse_mag = torch.mean(dmag**2)
                    mse_ang = torch.mean(dang**2)
                    rmse = torch.sqrt(mse_mag + mse_ang)
                    errs.append(float(rmse.item()))
            else:
                B = Vpred.shape[0]
                for b in range(B):
                    dmag = Vpred[b, :, 0] - Vref[b, :, 0]
                    dang = _angle_wrap_delta(Vpred[b, :, 1] - Vref[b, :, 1])
                    mse_mag = torch.mean(dmag**2)
                    mse_ang = torch.mean(dang**2)
                    rmse = torch.sqrt(mse_mag + mse_ang)
                    errs.append(float(rmse.item()))

    return np.asarray(errs, dtype=np.float64)


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError(f"Wilcoxon inputs must have same shape, got {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Wilcoxon inputs contain NaN or inf values")

    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0

    stat, p_value = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(p_value)


def _prepare_model(
    *,
    merged_raw: dict[str, Any],
    ckpt_path: str,
    device: torch.device,
    lora_r_override: int | None,
    lora_alpha_override: int | None,
    lora_dropout_override: float | None,
    lora_targets_override: list[str] | None,
) -> torch.nn.Module:
    model_name = str(get(merged_raw, ("model", "name"), "GNSMsg_EdgeSelfAttn"))
    d = int(get(merged_raw, ("model", "d"), 8))
    d_hi = int(get(merged_raw, ("model", "d_hi"), 32))
    num_attn_layers = int(get(merged_raw, ("model", "num_attn_layers"), 8))

    K = int(get(merged_raw, ("model", "K"), 40))
    gamma = float(get(merged_raw, ("model", "gamma"), 0.9))
    vlimit = bool(get(merged_raw, ("model", "vlimit"), True))
    use_armijo = bool(get(merged_raw, ("model", "use_armijo"), True))
    dtheta_max = float(get(merged_raw, ("model", "DthetaMax"), 0.3))
    dvm_frac = float(get(merged_raw, ("model", "DvmFrac"), 0.1))

    model = create_model(
        model_name=model_name,
        d=d,
        d_hi=d_hi,
        K=K,
        pinn=bool(get(merged_raw, ("flags", "pinn"), True)),
        gamma=gamma,
        v_limit=vlimit,
        use_armijo=use_armijo,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers,
        device=device,
    )

    sd = _load_state_dict(ckpt_path)
    is_lora_ckpt = _looks_like_lora_state_dict(sd)

    if is_lora_ckpt:
        lora_r_cfg = int(get(merged_raw, ("peft", "lora_r"), 8))
        lora_alpha_cfg = int(get(merged_raw, ("peft", "lora_alpha"), 16))
        lora_dropout_cfg = float(get(merged_raw, ("peft", "lora_dropout"), 0.0))
        lora_target_modules_cfg = list(get(merged_raw, ("peft", "lora_target_modules"), ["q", "k", "v", "out"]))

        lora_r = int(lora_r_override) if lora_r_override is not None else int(lora_r_cfg)
        lora_alpha = int(lora_alpha_override) if lora_alpha_override is not None else int(lora_alpha_cfg)
        lora_dropout = float(lora_dropout_override) if lora_dropout_override is not None else float(lora_dropout_cfg)
        lora_target_modules = list(lora_targets_override) if lora_targets_override is not None else list(lora_target_modules_cfg)

        detected_rank = _detect_lora_rank_from_state_dict(sd)
        if detected_rank is not None and lora_r_override is None and int(lora_r) != int(detected_rank):
            log.warning(
                "LoRA rank mismatch: config/args r=%s but checkpoint appears to be r=%s. Using checkpoint r=%s for loading.",
                int(lora_r),
                int(detected_rank),
                int(detected_rank),
            )
            lora_r = int(detected_rank)

        log.info("Applying LoRA wrappers (r=%s, alpha=%s)", lora_r, lora_alpha)
        apply_lora_to_linear_modules(
            model,
            target_module_names=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    model.load_state_dict(sd, strict=True)
    return model


def _load_merged_config(*, base_path: str, scenario_path: str | None) -> tuple[dict[str, Any], str]:
    base_raw = _ensure_mapping(load_yaml_config(str(base_path)), what="base config")
    merged_raw = dict(base_raw)
    label = str(base_path)

    if scenario_path:
        scenario_raw = _ensure_mapping(load_yaml_config(str(scenario_path)), what="scenario config")
        deep_update(merged_raw, scenario_raw)
        label = f"{base_path} + {scenario_path}"

    return merged_raw, label


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        merged_raw = _ensure_mapping(load_yaml_config(args.config), what="config")
        cfg_path_label = str(args.config)
        merged_by_method = None
    else:
        if not args.base_config:
            raise SystemExit("--base is required when --config is not provided")
        merged_by_method = {}
        # Optional per-method scenario overrides
        if args.scenario_full or args.scenario_config:
            merged_by_method["full_ft"] = _load_merged_config(
                base_path=str(args.base_config),
                scenario_path=str(args.scenario_full or args.scenario_config),
            )
        if args.scenario_head or args.scenario_config:
            merged_by_method["head_only"] = _load_merged_config(
                base_path=str(args.base_config),
                scenario_path=str(args.scenario_head or args.scenario_config),
            )
        if args.scenario_lora or args.scenario_config:
            merged_by_method["lora_only"] = _load_merged_config(
                base_path=str(args.base_config),
                scenario_path=str(args.scenario_lora or args.scenario_config),
            )
        if args.scenario_lora_head or args.scenario_config:
            merged_by_method["lora_head"] = _load_merged_config(
                base_path=str(args.base_config),
                scenario_path=str(args.scenario_lora_head or args.scenario_config),
            )

        # If no scenario passed at all, still load base for all methods on-demand
        if not merged_by_method:
            merged_by_method["full_ft"] = _load_merged_config(
                base_path=str(args.base_config),
                scenario_path=None,
            )
        merged_raw, cfg_path_label = merged_by_method.get("full_ft")

    if args.parquet is not None:
        parquet_paths = env_expand_paths(list(args.parquet))
    else:
        parquet_paths = env_expand_paths(as_path_list(get(merged_raw, ("data", "parquet_paths"), [])))

    if not parquet_paths:
        raise SystemExit("No parquet paths provided. Use --parquet or set data.parquet_paths in config.")

    seed_cfg = int(get(merged_raw, ("run", "seed"), 42))
    seed = int(args.seed) if args.seed is not None else int(seed_cfg)

    split_mode = str(get(merged_raw, ("split", "mode"), "equal3"))
    train_ratio = float(get(merged_raw, ("split", "train_ratio"), 0.8))
    valid_ratio = float(get(merged_raw, ("split", "valid_ratio"), 0.1))

    per_unit = bool(get(merged_raw, ("data", "per_unit"), True))
    pinn = bool(get(merged_raw, ("flags", "pinn"), True))
    block_diag = bool(get(merged_raw, ("flags", "block_diag"), True))

    cfg_train_bs = int(get(merged_raw, ("train", "batch_size"), 16))
    eval_batch_size = int(args.batch_size) if args.batch_size is not None else int(min(cfg_train_bs, 32))
    if eval_batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    if args.device:
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "significance_tests" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building dataloaders for evaluation (%d parquet files)", len(parquet_paths))
    splits = build_dataloaders(
        parquet_paths=parquet_paths,
        per_unit=per_unit,
        device=device,
        batch_size=eval_batch_size,
        block_diag=block_diag,
        seed=seed,
        split_mode=split_mode,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    log.info("Dataset sizes | train %s  valid %s  test %s", splits.n_train, splits.n_val, splits.n_test)

    methods: dict[str, str] = {
        "full_ft": args.full_ckpt,
    }
    if args.head_ckpt:
        methods["head_only"] = args.head_ckpt
    if args.lora_ckpt:
        methods["lora_only"] = args.lora_ckpt
    if args.lora_head_ckpt:
        methods["lora_head"] = args.lora_head_ckpt

    errors: dict[str, np.ndarray] = {}
    method_cfg_labels: dict[str, str] = {}
    for name, ckpt in methods.items():
        log.info("Evaluating %s -> %s", name, ckpt)
        if args.config:
            merged_raw_method = merged_raw
            cfg_label_method = cfg_path_label
        else:
            if merged_by_method is None:
                merged_raw_method, cfg_label_method = merged_raw, cfg_path_label
            else:
                if name in merged_by_method:
                    merged_raw_method, cfg_label_method = merged_by_method[name]
                else:
                    merged_raw_method, cfg_label_method = _load_merged_config(
                        base_path=str(args.base_config),
                        scenario_path=args.scenario_config,
                    )
        method_cfg_labels[name] = cfg_label_method
        pinn_method = bool(get(merged_raw_method, ("flags", "pinn"), True))
        block_diag_method = bool(get(merged_raw_method, ("flags", "block_diag"), True))
        if block_diag_method != block_diag:
            log.warning(
                "Method %s uses block_diag=%s, but dataloader was built with block_diag=%s. "
                "Evaluation will use the dataloader setting.",
                name,
                block_diag_method,
                block_diag,
            )
        model = _prepare_model(
            merged_raw=merged_raw_method,
            ckpt_path=str(ckpt),
            device=device,
            lora_r_override=args.lora_r,
            lora_alpha_override=args.lora_alpha,
            lora_dropout_override=args.lora_dropout,
            lora_targets_override=args.lora_target_modules,
        )
        errs = _per_sample_rmse(
            model=model,
            test_loader=splits.test_loader,
            device=device,
            pinn=pinn_method,
            block_diag=block_diag,
        )
        errors[name] = errs
        log.info("%s: collected %d per-sample errors", name, len(errs))

    if "full_ft" not in errors:
        raise SystemExit("Full FT errors missing; cannot run paired tests.")

    n_ref = len(errors["full_ft"])
    for k, v in errors.items():
        if len(v) != n_ref:
            raise SystemExit(f"Length mismatch for {k}: got {len(v)} vs full_ft {n_ref}")

    # Save per-sample errors CSV
    per_sample_df = pd.DataFrame({"sample_id": np.arange(n_ref, dtype=int)})
    for name, arr in errors.items():
        per_sample_df[f"rmse_{name}"] = arr

    per_sample_path = out_dir / "per_sample_errors.csv"
    per_sample_df.to_csv(per_sample_path, index=False)

    # Paired Wilcoxon tests vs Full FT
    summary_rows: list[dict[str, Any]] = []
    full = errors["full_ft"]

    for name, arr in errors.items():
        mean_rmse = float(np.mean(arr))
        median_rmse = float(np.median(arr))
        row = {
            "method": name,
            "n_samples": int(len(arr)),
            "mean_rmse": mean_rmse,
            "median_rmse": median_rmse,
        }
        if name != "full_ft":
            stat, p_value = _safe_wilcoxon(arr, full)
            p_value_str = f"{p_value:.17g}"
            p_lower_bound = float(np.nextafter(0, 1, dtype=np.float64))
            p_value_floor = p_value if p_value > 0.0 else p_lower_bound
            neg_log10_p = float(-math.log10(p_value_floor))
            row.update(
                {
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "p_value_str": p_value_str,
                    "neg_log10_p": neg_log10_p,
                }
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "significance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "config": cfg_path_label,
        "parquet_paths": list(parquet_paths),
        "seed": int(seed),
        "split": {"mode": split_mode, "train_ratio": train_ratio, "valid_ratio": valid_ratio},
        "eval_batch_size": int(eval_batch_size),
        "methods": methods,
        "method_configs": method_cfg_labels,
        "device": str(device),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Saved per-sample CSV to %s", per_sample_path)
    log.info("Saved summary CSV to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
