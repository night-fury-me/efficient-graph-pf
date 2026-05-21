#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

# When invoked as `python scripts/eval_forgetting.py`, Python puts `scripts/` at the
# front of sys.path. Prepend repo root so `import train...` resolves to the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.config_loader import as_path_list, deep_update, env_expand_paths, get, load_yaml_config
from train.data import build_dataloaders
from train.loop import evaluate_test
from train.modeling import create_model
from train.mlflow_utils import add_basic_tags, log_run_artifacts, mlflow_run
from train.peft_utils import apply_lora_to_linear_modules


log = logging.getLogger("eval_forgetting")


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
    # Torch can store non-tensor values, but state_dict() should be tensors.
    return payload  # type: ignore[return-value]


def _looks_like_lora_state_dict(sd: Mapping[str, Any]) -> bool:
    for k in sd.keys():
        if ".lora_A" in str(k) or ".lora_B" in str(k):
            return True
    return False


def _detect_lora_rank_from_state_dict(sd: Mapping[str, Any]) -> int | None:
    """Infer LoRA rank r from saved adapter tensors.

    We look at the first dimension of every *.lora_A parameter, expecting shape (r, in_features).
    Returns None if no LoRA params found.
    Raises if multiple ranks are found.
    """

    ranks: set[int] = set()
    for k, v in sd.items():
        if ".lora_A" not in str(k):
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
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
            "Evaluate an adapted checkpoint on the source domain to measure catastrophic forgetting. "
            "Supports both full fine-tuning checkpoints (plain nn.Linear) and LoRA checkpoints (LoRALinear)."
        )
    )

    cfg = p.add_mutually_exclusive_group(required=True)
    cfg.add_argument(
        "--config",
        default=None,
        help=(
            "Single YAML config file (already merged). "
            "For LoRA checkpoints, must include the `peft` section used during LoRA training (r/alpha/targets)."
        ),
    )
    cfg.add_argument(
        "--base",
        "--base-config",
        dest="base_config",
        default=None,
        help=(
            "Base YAML config (e.g., configs/default.yaml). Use with --scenario for overlay-style configs."
        ),
    )

    p.add_argument(
        "--scenario",
        "--scenario-config",
        dest="scenario_config",
        default=None,
        help=(
            "Optional scenario overlay YAML (e.g., configs/scenarios/lora_ft_hv.yaml). "
            "Deep-merged on top of --base."
        ),
    )

    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint to evaluate (e.g., results/.../ckpt/best.ckpt)",
    )

    p.add_argument(
        "--source-parquet",
        nargs="+",
        default=None,
        help=(
            "Parquet path(s) for the SOURCE domain (MV). Overrides data.parquet_paths in --config. "
            "Example: --source-parquet ./datasets/MVN_30000_...parquet ./datasets/MVN_30010_...parquet"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override seed used for the SOURCE train/val/test split. "
            "Use the same seed across methods to compare forgetting fairly."
        ),
    )

    p.add_argument(
        "--ckpt-type",
        choices=("auto", "full", "lora"),
        default="auto",
        help="How to interpret --ckpt. 'auto' detects LoRA by presence of lora_A/lora_B keys.",
    )

    p.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help=(
            "Override LoRA rank r used when loading a LoRA checkpoint. "
            "If omitted, will use config.peft.lora_r; if that mismatches the checkpoint, the script auto-detects r from the checkpoint and uses it."
        ),
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help=(
            "Override LoRA alpha used for evaluation scaling. Default: config.peft.lora_alpha. "
            "Note: alpha is not stored in the checkpoint state_dict in this repo, so ensure this matches training."
        ),
    )

    p.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., cuda, cuda:0, cpu). Default: auto-detect.",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Evaluation batch size. If omitted, uses a conservative default (min(config.train.batch_size, 32))."
        ),
    )

    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for metrics artifacts. Default: results/forgetting_evals/<timestamp>_<ckpt_stem>",
    )

    p.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging (overrides config.mlflow.enabled).",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (overrides config.mlflow.enabled).",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    base_raw: dict[str, Any] = {}
    scenario_raw: dict[str, Any] = {}

    if args.config:
        merged_raw = _ensure_mapping(load_yaml_config(args.config), what="config")
        cfg_path_label = str(args.config)
    else:
        if not args.base_config:
            raise SystemExit("--base is required when --config is not provided")
        base_raw = _ensure_mapping(load_yaml_config(str(args.base_config)), what="base config")
        merged_raw = dict(base_raw)
        cfg_path_label = str(args.base_config)

        if args.scenario_config:
            scenario_raw = _ensure_mapping(load_yaml_config(str(args.scenario_config)), what="scenario config")
            deep_update(merged_raw, scenario_raw)
            cfg_path_label = f"{args.base_config} + {args.scenario_config}"

    # Resolve source domain dataset paths.
    # If using base+scenario overlays and the scenario is an HV adaptation config,
    # default to *base* data.parquet_paths (MV) unless explicitly overridden.
    if args.source_parquet is not None:
        parquet_paths = env_expand_paths(list(args.source_parquet))
    else:
        if args.config:
            parquet_paths = env_expand_paths(as_path_list(get(merged_raw, ("data", "parquet_paths"), [])))
        else:
            parquet_paths = env_expand_paths(as_path_list(get(base_raw, ("data", "parquet_paths"), [])))

    if not parquet_paths:
        raise SystemExit("No parquet paths provided. Use --source-parquet or set data.parquet_paths in config.")

    # Config -> runtime fields
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

    # PEFT (for LoRA checkpoints)
    peft_enabled = bool(get(merged_raw, ("peft", "enabled"), False))
    lora_r_cfg = int(get(merged_raw, ("peft", "lora_r"), 8))
    lora_alpha_cfg = int(get(merged_raw, ("peft", "lora_alpha"), 16))
    lora_r = int(args.lora_r) if args.lora_r is not None else int(lora_r_cfg)
    lora_alpha = int(args.lora_alpha) if args.lora_alpha is not None else int(lora_alpha_cfg)
    lora_dropout = float(get(merged_raw, ("peft", "lora_dropout"), 0.0))
    lora_target_modules = list(get(merged_raw, ("peft", "lora_target_modules"), ["q", "k", "v", "out"]))

    # MLflow config
    cfg_mlflow_enabled = bool(get(merged_raw, ("mlflow", "enabled"), False))
    mlflow_tracking_uri = get(merged_raw, ("mlflow", "tracking_uri"), None)
    mlflow_experiment = str(get(merged_raw, ("mlflow", "experiment"), "ForgettingEval"))
    mlflow_artifact_location = get(merged_raw, ("mlflow", "artifact_location"), None)
    mlflow_artifact_path = str(get(merged_raw, ("mlflow", "artifact_path"), "run"))
    mlflow_strict = bool(get(merged_raw, ("mlflow", "strict"), True))

    if args.mlflow:
        mlflow_enabled = True
    elif args.no_mlflow:
        mlflow_enabled = False
    else:
        mlflow_enabled = cfg_mlflow_enabled

    ckpt_path = str(args.ckpt)
    sd = _load_state_dict(ckpt_path)

    detected_is_lora = _looks_like_lora_state_dict(sd)
    detected_rank = _detect_lora_rank_from_state_dict(sd) if detected_is_lora else None
    if args.ckpt_type == "auto":
        is_lora_ckpt = detected_is_lora
    elif args.ckpt_type == "lora":
        is_lora_ckpt = True
    else:
        is_lora_ckpt = False

    if is_lora_ckpt and not peft_enabled:
        log.warning(
            "Checkpoint looks like a LoRA checkpoint, but config.peft.enabled is false. "
            "Proceeding anyway (will apply LoRA wrappers using peft.* values from config)."
        )

    if is_lora_ckpt and detected_rank is not None and args.lora_r is None and int(lora_r) != int(detected_rank):
        log.warning(
            "LoRA rank mismatch: config/args r=%s but checkpoint appears to be r=%s. Using checkpoint r=%s for loading.",
            int(lora_r),
            int(detected_rank),
            int(detected_rank),
        )
        lora_r = int(detected_rank)

    # Device selection
    if args.device:
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_stem = Path(ckpt_path).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "forgetting_evals" / f"{timestamp}_{ckpt_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist a small run manifest
    manifest = {
        "config": cfg_path_label,
        "ckpt": str(ckpt_path),
        "ckpt_type": "lora" if is_lora_ckpt else "full",
        "detected_ckpt_is_lora": bool(detected_is_lora),
        "device": str(device),
        "seed": int(seed),
        "seed_from_config": int(seed_cfg),
        "seed_overridden": bool(args.seed is not None),
        "split": {"mode": split_mode, "train_ratio": train_ratio, "valid_ratio": valid_ratio},
        "eval_batch_size": int(eval_batch_size),
        "source_parquet_paths": list(parquet_paths),
        "peft": {
            "enabled_in_config": bool(peft_enabled),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "lora_target_modules": list(lora_target_modules),
            "lora_r_from_config": int(lora_r_cfg),
            "lora_alpha_from_config": int(lora_alpha_cfg),
            "lora_r_overridden": bool(args.lora_r is not None),
            "lora_alpha_overridden": bool(args.lora_alpha is not None),
            "lora_rank_detected_from_ckpt": int(detected_rank) if detected_rank is not None else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Building dataloaders for SOURCE domain (%d parquet files)", len(parquet_paths))
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
    log.info("SOURCE dataset sizes | train %s  valid %s  test %s", splits.n_train, splits.n_val, splits.n_test)

    log.info("Creating model: %s", model_name)
    model = create_model(
        model_name=model_name,
        d=d,
        d_hi=d_hi,
        K=K,
        pinn=pinn,
        gamma=gamma,
        v_limit=vlimit,
        use_armijo=use_armijo,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers,
        device=device,
    )

    if is_lora_ckpt:
        log.info("Applying LoRA wrappers before loading checkpoint (r=%s, alpha=%s)", lora_r, lora_alpha)
        wrapped = apply_lora_to_linear_modules(
            model,
            target_module_names=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        log.info("LoRA wrapped modules: %d", len(wrapped))

    log.info("Loading checkpoint: %s", ckpt_path)
    model.load_state_dict(sd, strict=True)

    log.info("Evaluating on SOURCE test split...")
    m = evaluate_test(
        model=model,
        test_loader=splits.test_loader,
        device=device,
        pinn=pinn,
        block_diag=block_diag,
        show_progress=True,
    )

    metrics = {
        "source/test/loss": float(m.loss),
        "source/test/rmse": float(m.rmse),
        "source/test/rmse_mag": float(m.rmse_mag),
        "source/test/rmse_ang_deg": float(m.rmse_ang_deg),
        "source/n_train": float(splits.n_train),
        "source/n_val": float(splits.n_val),
        "source/n_test": float(splits.n_test),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Also write a tiny CSV for easy aggregation
    (out_dir / "metrics.csv").write_text(
        "key,value\n" + "\n".join([f"{k},{v}" for k, v in metrics.items()]) + "\n",
        encoding="utf-8",
    )

    # Optional MLflow logging
    tags = add_basic_tags(repo_root=_REPO_ROOT)
    tags.update(
        {
            "eval_type": "forgetting",
            "domain": "source",
            "ckpt_type": "lora" if is_lora_ckpt else "full",
            "ckpt": str(ckpt_path),
            "seed": str(seed),
        }
    )

    run_name = f"forgetting_{Path(ckpt_path).stem}"
    with mlflow_run(
        enabled=mlflow_enabled,
        strict=mlflow_strict,
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        artifact_location=mlflow_artifact_location,
        run_name=run_name,
        tags=tags,
    ) as mlf:
        if mlf is not None:
            for k, v in metrics.items():
                try:
                    mlf.log_metric(k, float(v))
                except Exception:
                    pass
            try:
                mlf.log_artifact(str(out_dir / "manifest.json"), artifact_path="eval")
                mlf.log_artifact(str(out_dir / "metrics.json"), artifact_path="eval")
                mlf.log_artifact(str(out_dir / "metrics.csv"), artifact_path="eval")
            except Exception:
                pass

            # If you want the whole folder in artifacts, do it once.
            try:
                log_run_artifacts(mlflow=mlf, run_dir=out_dir, artifact_path=os.path.join(mlflow_artifact_path, "forgetting_eval"))
            except Exception:
                pass

    log.info("Done. Metrics written to %s", out_dir)
    log.info(
        "SOURCE test | rmse=%.6g (mag=%.6g, ang=%.6g°)",
        metrics["source/test/rmse"],
        metrics["source/test/rmse_mag"],
        metrics["source/test/rmse_ang_deg"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
