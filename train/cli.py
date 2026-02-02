from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

from .config_loader import as_path_list, env_expand_paths, get, load_yaml_config


DEFAULTS = {
    "PINN": False,
    "BLOCK_DIAG": False,
    "NORMALIZE": False,
    "PER_UNIT": False,
    "float64": False,
    "mode": "train_test",
    "mag_ang_mse": False,
    "model": "GNSMsg_EdgeSelfAttn",
    "d": 4,
    "d_hi": 16,
    "num_attn_layers": 1,
    "K": 40,
    "gamma": 0.9,
    "use_armijo": False,
    "vlimit": False,
    "DthetaMax": 0.3,
    "DvmFrac": 0.1,
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "split_mode": "ratio",  # ratio | equal3
    "ADJ_MODE": "cplx",
    "weight_init": "sd0.02",
    "bias_init": 0.0,
    "weight_decay": 1e-3,
    "lr_scheduler": "default",
    "cosineRestartEpoch": 20,
    "BATCH": 16,
    "EPOCHS": 20,
    "LR": 1e-4,
    "VAL_EVERY": 1,
    "PARQUET": ["./datasets/HVN_15000_NR_plain_4_to_32_buses.parquet"],
    "seed_value": 42,

    # MLflow
    "mlflow": False,
    "mlflow_tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "SimpleGNN",
}


def build_parser(*, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training script with configurable hyperparameters"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. config.yaml). CLI args override YAML.",
    )

    dflt = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--PINN", action="store_true", default=dflt, help="Enable Physics-Informed Neural Networks")
    parser.add_argument("--BLOCK_DIAG", action="store_true", default=dflt, help="Use block diagonal mode")
    parser.add_argument("--NORMALIZE", action="store_true", default=dflt, help="Enable normalization")
    parser.add_argument("--PER_UNIT", action="store_true", default=dflt, help="Use per-unit scaling")
    parser.add_argument("--float64", action="store_true", default=dflt, help="Use float64")
    parser.add_argument(
        "--mode",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mode"]),
        help="train_valid_test | train | valid | test",
    )
    parser.add_argument(
        "--mag_ang_mse",
        action="store_true",
        default=dflt,
        help="normalised |V| + wrapped-angle loss",
    )

    parser.add_argument("--model", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["model"]), help="Model name")
    parser.add_argument("--d", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["d"]), help="model input dim")
    parser.add_argument("--d_hi", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["d_hi"]), help="model hidden dim")
    parser.add_argument("--num_attn_layers", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["num_attn_layers"]), help="number of attention layers")

    parser.add_argument("--K", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["K"]), help="K")
    parser.add_argument("--gamma", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["gamma"]), help="phys_loss decay over K")
    parser.add_argument("--use_armijo", action="store_true", default=dflt, help="use_armijo")
    parser.add_argument("--vlimit", action="store_true", default=dflt, help="vlimit disabled")
    parser.add_argument("--DthetaMax", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["DthetaMax"]), help="DthetaMax")
    parser.add_argument("--DvmFrac", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["DvmFrac"]), help="DvmFrac")
    parser.add_argument("--train_ratio", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["train_ratio"]), help="train_ratio")
    parser.add_argument("--valid_ratio", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["valid_ratio"]), help="valid_ratio")
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=("ratio", "equal3"),
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["split_mode"]),
        help="Dataset split strategy: ratio (train_ratio/valid_ratio) or equal3 (1/3 each; remainder distributed)",
    )

    parser.add_argument("--ADJ_MODE", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["ADJ_MODE"]), help="Adjacency mode: real | cplx | other")

    parser.add_argument("--weight_init", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["weight_init"]), help="Weight initialization method (None, He, sd0.02)")
    parser.add_argument("--bias_init", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["bias_init"]), help="Bias initialization value")
    parser.add_argument("--weight_decay", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["weight_decay"]), help="Weight decay rate")

    parser.add_argument("--lr_scheduler", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lr_scheduler"]), help="CosineAnnealingLR")
    parser.add_argument("--cosineRestartEpoch", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["cosineRestartEpoch"]), help="cosineRestartEpoch")

    parser.add_argument("--BATCH", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["BATCH"]), help="Batch size")
    parser.add_argument("--EPOCHS", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["EPOCHS"]), help="Number of training epochs")
    parser.add_argument("--LR", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["LR"]), help="Learning rate")
    parser.add_argument("--VAL_EVERY", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["VAL_EVERY"]), help="Validation frequency (in epochs)")

    parser.add_argument(
        "--PARQUET",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["PARQUET"]),
        help="Path to Parquet data file(s)",
    )

    # MLflow
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow"]),
        help="Enable MLflow logging",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_tracking_uri"]),
        help="MLflow tracking URI (e.g. sqlite:///mlflow.db, file:./mlruns, http://localhost:5000)",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_experiment"]),
        help="MLflow experiment name",
    )

    parser.add_argument("--seed_value", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["seed_value"]), help="Random seed")
    return parser


def _short_dataset_name(parquet_paths: List[str]) -> str:
    parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in parquet_paths]
    shortened_names = ["_".join(name.split("_")[:3]) for name in parquet_filenames]
    return "_and_".join(shortened_names)


@dataclass(frozen=True)
class TrainConfig:
    # Run
    runname: str
    seed: int
    mode: str

    # Data
    parquet_paths: List[str]
    split_mode: str
    train_ratio: float
    valid_ratio: float

    # MLflow
    mlflow: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment: str

    # Model
    model_name: str
    d: int
    d_hi: int
    num_attn_layers: int
    K: int
    gamma: float
    vlimit: bool
    use_armijo: bool
    dtheta_max: float
    dvm_frac: float

    # Training
    batch_size: int
    epochs: int
    lr: float
    val_every: int
    weight_init: str
    bias_init: float
    weight_decay: float
    lr_scheduler: str
    cosine_restart_epoch: int

    # Feature flags
    pinn: bool
    block_diag: bool
    normalize: bool
    per_unit: bool
    mag_ang_mse: bool


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    # Backwards-compatible path: args already includes defaults.
    dataset_name = _short_dataset_name(args.PARQUET)
    split_tag = "eq3" if getattr(args, "split_mode", "ratio") == "equal3" else f"tr{args.train_ratio:g}"
    model_tag = str(args.model).replace("GNSMsg_", "")
    runname = f"{dataset_name}_{model_tag}_k{args.K}_d{args.d}_h{args.d_hi}_ep{args.EPOCHS}_{split_tag}"
    return TrainConfig(
        runname=runname,
        seed=int(args.seed_value),
        mode=str(args.mode),
        parquet_paths=list(args.PARQUET),
        split_mode=str(getattr(args, "split_mode", DEFAULTS["split_mode"])),
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
        mlflow=bool(getattr(args, "mlflow", False)),
        mlflow_tracking_uri=getattr(args, "mlflow_tracking_uri", None),
        mlflow_experiment=str(getattr(args, "mlflow_experiment", DEFAULTS["mlflow_experiment"])),
        model_name=str(args.model),
        d=int(args.d),
        d_hi=int(args.d_hi),
        num_attn_layers=int(args.num_attn_layers),
        K=int(args.K),
        gamma=float(args.gamma),
        vlimit=bool(args.vlimit),
        use_armijo=bool(args.use_armijo),
        dtheta_max=float(getattr(args, "DthetaMax", DEFAULTS["DthetaMax"])),
        dvm_frac=float(getattr(args, "DvmFrac", DEFAULTS["DvmFrac"])),
        batch_size=int(args.BATCH),
        epochs=int(args.EPOCHS),
        lr=float(args.LR),
        val_every=int(args.VAL_EVERY),
        weight_init=str(args.weight_init),
        bias_init=float(args.bias_init),
        weight_decay=float(args.weight_decay),
        lr_scheduler=str(args.lr_scheduler),
        cosine_restart_epoch=int(args.cosineRestartEpoch),
        pinn=bool(args.PINN),
        block_diag=bool(args.BLOCK_DIAG),
        normalize=bool(args.NORMALIZE),
        per_unit=bool(args.PER_UNIT),
        mag_ang_mse=bool(args.mag_ang_mse),
    )


def parse_train_config(argv: list[str] | None = None) -> tuple[TrainConfig, str | None]:
    """Parse CLI + optional YAML config.

    Returns (TrainConfig, config_path).
    """
    # Pass 1: detect --config without applying defaults for other args.
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default=None)
    probe_ns, remaining = config_probe.parse_known_args(argv)

    if probe_ns.config:
        raw = load_yaml_config(probe_ns.config)

        # Pass 2: only parse explicit CLI overrides.
        parser = build_parser(suppress_defaults=True)
        args = parser.parse_args(remaining)

        # Merge order: hard defaults -> YAML -> CLI overrides
        merged: dict[str, object] = dict(DEFAULTS)

        # YAML (nested)
        merged["mode"] = get(raw, ("run", "mode"), merged["mode"])
        merged["seed_value"] = get(raw, ("run", "seed"), merged["seed_value"])

        merged["train_ratio"] = get(raw, ("split", "train_ratio"), merged["train_ratio"])
        merged["valid_ratio"] = get(raw, ("split", "valid_ratio"), merged["valid_ratio"])
        merged["split_mode"] = get(raw, ("split", "mode"), merged["split_mode"])

        parquet_paths = env_expand_paths(as_path_list(get(raw, ("data", "parquet_paths"), merged["PARQUET"])))
        merged["PARQUET"] = parquet_paths
        merged["PER_UNIT"] = bool(get(raw, ("data", "per_unit"), merged["PER_UNIT"]))

        merged["PINN"] = bool(get(raw, ("flags", "pinn"), merged["PINN"]))
        merged["BLOCK_DIAG"] = bool(get(raw, ("flags", "block_diag"), merged["BLOCK_DIAG"]))
        merged["NORMALIZE"] = bool(get(raw, ("flags", "normalize"), merged["NORMALIZE"]))
        merged["mag_ang_mse"] = bool(get(raw, ("flags", "mag_ang_mse"), merged["mag_ang_mse"]))
        merged["float64"] = bool(get(raw, ("flags", "float64"), merged["float64"]))

        merged["model"] = get(raw, ("model", "name"), merged["model"])
        merged["d"] = int(get(raw, ("model", "d"), merged["d"]))
        merged["d_hi"] = int(get(raw, ("model", "d_hi"), merged["d_hi"]))
        merged["num_attn_layers"] = int(get(raw, ("model", "num_attn_layers"), merged["num_attn_layers"]))
        merged["K"] = int(get(raw, ("model", "K"), merged["K"]))
        merged["gamma"] = float(get(raw, ("model", "gamma"), merged["gamma"]))
        merged["vlimit"] = bool(get(raw, ("model", "vlimit"), merged["vlimit"]))
        merged["use_armijo"] = bool(get(raw, ("model", "use_armijo"), merged["use_armijo"]))
        merged["DthetaMax"] = float(get(raw, ("model", "DthetaMax"), merged["DthetaMax"]))
        merged["DvmFrac"] = float(get(raw, ("model", "DvmFrac"), merged["DvmFrac"]))

        merged["BATCH"] = int(get(raw, ("train", "batch_size"), merged["BATCH"]))
        merged["EPOCHS"] = int(get(raw, ("train", "epochs"), merged["EPOCHS"]))
        merged["LR"] = float(get(raw, ("train", "lr"), merged["LR"]))
        merged["VAL_EVERY"] = int(get(raw, ("train", "val_every"), merged["VAL_EVERY"]))

        merged["weight_init"] = get(raw, ("optimizer", "weight_init"), merged["weight_init"])
        merged["bias_init"] = float(get(raw, ("optimizer", "bias_init"), merged["bias_init"]))
        merged["weight_decay"] = float(get(raw, ("optimizer", "weight_decay"), merged["weight_decay"]))
        merged["lr_scheduler"] = get(raw, ("optimizer", "lr_scheduler"), merged["lr_scheduler"])
        merged["cosineRestartEpoch"] = int(get(raw, ("optimizer", "cosine_restart_epoch"), merged["cosineRestartEpoch"]))

        merged["ADJ_MODE"] = get(raw, ("misc", "adj_mode"), merged["ADJ_MODE"])

        # MLflow
        merged["mlflow"] = bool(get(raw, ("mlflow", "enabled"), merged["mlflow"]))
        merged["mlflow_tracking_uri"] = get(raw, ("mlflow", "tracking_uri"), merged["mlflow_tracking_uri"])
        merged["mlflow_experiment"] = str(get(raw, ("mlflow", "experiment"), merged["mlflow_experiment"]))

        # CLI overrides (only what the user explicitly provided)
        for k, v in vars(args).items():
            if k == "config":
                continue
            merged[k] = v

        # Optional explicit run name (YAML). If omitted/null, auto-generate from the
        # final merged config (after CLI overrides).
        runname = get(raw, ("run", "name"), None)
        if runname:
            merged_runname = str(runname)
        else:
            dataset_name = _short_dataset_name(merged["PARQUET"])  # type: ignore[arg-type]
            split_tag = "eq3" if str(merged.get("split_mode", "ratio")) == "equal3" else f"tr{float(merged['train_ratio']):g}"
            model_tag = str(merged["model"]).replace("GNSMsg_", "")
            merged_runname = f"{dataset_name}_{model_tag}_k{merged['K']}_d{merged['d']}_h{merged['d_hi']}_ep{merged['EPOCHS']}_{split_tag}"

        cfg = TrainConfig(
            runname=merged_runname,
            seed=int(merged["seed_value"]),
            mode=str(merged["mode"]),
            parquet_paths=list(merged["PARQUET"]),
            split_mode=str(merged.get("split_mode", DEFAULTS["split_mode"])),
            train_ratio=float(merged["train_ratio"]),
            valid_ratio=float(merged["valid_ratio"]),
            mlflow=bool(merged.get("mlflow", False)),
            mlflow_tracking_uri=(str(merged["mlflow_tracking_uri"]) if merged.get("mlflow_tracking_uri") else None),
            mlflow_experiment=str(merged.get("mlflow_experiment", DEFAULTS["mlflow_experiment"])),
            model_name=str(merged["model"]),
            d=int(merged["d"]),
            d_hi=int(merged["d_hi"]),
            num_attn_layers=int(merged["num_attn_layers"]),
            K=int(merged["K"]),
            gamma=float(merged["gamma"]),
            vlimit=bool(merged.get("vlimit", False)),
            use_armijo=bool(merged.get("use_armijo", False)),
            dtheta_max=float(merged.get("DthetaMax", DEFAULTS["DthetaMax"])),
            dvm_frac=float(merged.get("DvmFrac", DEFAULTS["DvmFrac"])),
            batch_size=int(merged["BATCH"]),
            epochs=int(merged["EPOCHS"]),
            lr=float(merged["LR"]),
            val_every=int(merged["VAL_EVERY"]),
            weight_init=str(merged["weight_init"]),
            bias_init=float(merged["bias_init"]),
            weight_decay=float(merged["weight_decay"]),
            lr_scheduler=str(merged["lr_scheduler"]),
            cosine_restart_epoch=int(merged["cosineRestartEpoch"]),
            pinn=bool(merged.get("PINN", False)),
            block_diag=bool(merged.get("BLOCK_DIAG", False)),
            normalize=bool(merged.get("NORMALIZE", False)),
            per_unit=bool(merged.get("PER_UNIT", False)),
            mag_ang_mse=bool(merged.get("mag_ang_mse", False)),
        )
        return cfg, str(probe_ns.config)

    # No config.yaml: preserve legacy CLI defaults.
    parser = build_parser(suppress_defaults=False)
    args = parser.parse_args(argv)
    return config_from_args(args), None
