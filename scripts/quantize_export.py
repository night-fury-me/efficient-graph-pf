from __future__ import annotations

import argparse
import sys
import csv
import json
import os
import time
from itertools import islice, cycle
from pathlib import Path
from typing import Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.cli import parse_train_config
from train.data import build_dataloaders
from train.logger import configure_logging, log
from train.logging_utils import ensure_run_dirs, make_run_paths
from train.mlflow_utils import add_basic_tags, log_params_safe, log_run_artifacts, mlflow_run, snapshot_code
from train.modeling import count_parameters, create_model
from train.peft_utils import apply_lora_to_linear_modules, merge_lora_weights
from train.loop import evaluate_test
from train.run_naming import make_run_id, make_run_slug, safe_param_dict


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Dynamic quantization export + benchmark")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (state_dict)")
    parser.add_argument("--out", type=str, required=True, help="Path to save quantized model state_dict")
    parser.add_argument("--config", type=str, default=None, help="Training YAML config used for the model")
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation on test set")
    parser.add_argument("--bench_warmup", type=int, default=20, help="Warmup iterations for CPU benchmark")
    parser.add_argument("--bench_iters", type=int, default=200, help="Timed iterations for CPU benchmark")
    parser.add_argument("--bench_max_batches", type=int, default=64, help="Max batches to sample for benchmark")
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=None,
        help="Override MLflow experiment name (optional)",
    )
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def _load_state_dict(ckpt_path: str) -> dict:
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint does not contain a state_dict dictionary")
    return sd


def _model_forward(model, batch: dict, *, device: torch.device, pinn: bool, block_diag: bool):
    n_nodes_per_graph = batch["sizes"].to(device) if block_diag else None
    bus_type = batch["bus_type"].to(device)
    Line = batch["Lines_connected"].to(device)
    Y_raw = batch.get("Ybus", None)
    Y = Y_raw.to(device, non_blocking=True) if isinstance(Y_raw, torch.Tensor) else None
    Ys = batch["Y_Lines"].to(device)
    Yc = batch["Y_C_Lines"].to(device)
    Sstart = batch["S_start"].to(device)
    Vstart = batch["V_start"].to(device)

    if pinn:
        Vpred, _ = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
    else:
        Vpred = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph)
    return Vpred


def _benchmark_inference(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    warmup: int,
    iters: int,
    max_batches: int,
) -> dict[str, float]:
    model.eval()

    batches = list(islice(loader, max_batches))
    if not batches:
        raise RuntimeError("Benchmark requested but loader produced zero batches")

    # Warmup
    with torch.no_grad():
        for i in range(max(0, int(warmup))):
            batch = batches[i % len(batches)]
            _model_forward(model, batch, device=device, pinn=pinn, block_diag=block_diag)

    times_ms: list[float] = []
    with torch.no_grad():
        for batch in islice(cycle(batches), int(iters)):
            t0 = time.perf_counter()
            _model_forward(model, batch, device=device, pinn=pinn, block_diag=block_diag)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    p50 = times_ms[int(0.50 * (len(times_ms) - 1))]
    p90 = times_ms[int(0.90 * (len(times_ms) - 1))]
    mean = sum(times_ms) / len(times_ms)
    return {
        "latency_p50_ms": float(p50),
        "latency_p90_ms": float(p90),
        "latency_mean_ms": float(mean),
    }


def _apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    try:
        import torchao.quantization as tq  # type: ignore

        if hasattr(tq, "quantize_"):
            try:
                return tq.quantize_(model, mode="int8_dynamic", inplace=False)
            except Exception:
                try:
                    return tq.quantize_(model, inplace=False)
                except Exception as e:
                    log.warning("torchao.quantize_ failed; falling back to torch.ao.quantization (%s)", e)
    except Exception as e:
        log.warning("torchao not available for quantization (%s); using torch.ao.quantization", e)

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _metric_deltas(base: dict[str, float], quant: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for k, base_val in base.items():
        if k not in quant:
            continue
        cur = float(quant[k])
        base_v = float(base_val)
        if base_v == 0.0:
            continue
        pct = 100.0 * ((cur - base_v) / base_v)
        if cur == 0.0:
            factor_x = float("-inf") if pct < 0 else float("inf")
        else:
            factor_x = -abs(base_v / cur) if cur < base_v else abs(cur / base_v)
        deltas[f"{k}/change_pct"] = float(pct)
        deltas[f"{k}/factor_x"] = float(factor_x)
    return deltas


def main(argv: list[str] | None = None) -> int:
    args, overrides = _parse_args(argv)

    # Parse training config (YAML + optional CLI overrides for data/model params)
    train_argv: list[str] = []
    if args.config:
        train_argv += ["--config", args.config]
    train_argv += overrides
    cfg, config_path = parse_train_config(train_argv)

    repo_root = Path(__file__).resolve().parents[1]

    run_slug = make_run_slug(
        parquet_paths=cfg.parquet_paths,
        model_name=cfg.model_name,
        K=cfg.K,
        d=cfg.d,
        d_hi=cfg.d_hi,
        pinn=cfg.pinn,
        block_diag=cfg.block_diag,
        per_unit=cfg.per_unit,
        split_mode=cfg.split_mode,
    )
    run_slug = f"{run_slug}_ptq"
    run_id = make_run_id(run_slug=run_slug)
    run_name = f"{run_id}_{run_slug}"

    paths = make_run_paths(run_id=run_id, base_dir="./results/quant")
    ensure_run_dirs(paths)
    log_file = str(Path(paths.run_dir) / "quantize.log")
    configure_logging(log_file=log_file)

    log.info("Quantization run: %s", run_name)
    log.info("Checkpoint: %s", args.ckpt)
    if config_path:
        log.info("Config: %s", config_path)

    device = torch.device("cpu")
    log.info("Using device: %s", device)

    splits = build_dataloaders(
        parquet_paths=cfg.parquet_paths,
        per_unit=cfg.per_unit,
        device=device,
        batch_size=cfg.batch_size,
        block_diag=cfg.block_diag,
        seed=cfg.seed,
        split_mode=cfg.split_mode,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
    )
    log.info("Dataset sizes | train %s  valid %s  test %s", splits.n_train, splits.n_val, splits.n_test)

    model_name = str(cfg.model_name).split(".")[-1]
    model = create_model(
        model_name=model_name,
        d=cfg.d,
        d_hi=cfg.d_hi,
        K=cfg.K,
        pinn=cfg.pinn,
        gamma=cfg.gamma,
        v_limit=cfg.vlimit,
        use_armijo=cfg.use_armijo,
        dtheta_max=cfg.dtheta_max,
        dvm_frac=cfg.dvm_frac,
        num_attn_layers=cfg.num_attn_layers,
        device=device,
    )

    # Optional LoRA wrapper
    if bool(getattr(cfg, "peft", False)):
        wrapped = apply_lora_to_linear_modules(
            model,
            target_module_names=list(getattr(cfg, "lora_target_modules", ["q", "k", "v", "out"])),
            r=int(getattr(cfg, "lora_r", 8)),
            alpha=int(getattr(cfg, "lora_alpha", 16)),
            dropout=float(getattr(cfg, "lora_dropout", 0.0)),
        )
        log.info("Applied LoRA to %d Linear modules", len(wrapped))

    state_dict = _load_state_dict(args.ckpt)
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded checkpoint into model")

    total_params = int(count_parameters(model))
    log.info("Total parameters: %s", total_params)

    base_metrics: dict[str, float] = {}
    quant_metrics: dict[str, float] = {}
    base_bench: dict[str, float] = {}
    quant_bench: dict[str, float] = {}

    if not args.no_eval:
        base_eval = evaluate_test(
            model=model,
            test_loader=splits.test_loader,
            device=device,
            pinn=cfg.pinn,
            block_diag=cfg.block_diag,
            show_progress=False,
        )
        base_metrics = {
            "loss": float(base_eval.loss),
            "rmse": float(base_eval.rmse),
            "rmse_mag": float(base_eval.rmse_mag),
            "rmse_ang_deg": float(base_eval.rmse_ang_deg),
        }
        log.info(
            "Base FP32 test | loss %.6g rmse %.6g (mag %.6g, ang %.6g)",
            base_metrics["loss"],
            base_metrics["rmse"],
            base_metrics["rmse_mag"],
            base_metrics["rmse_ang_deg"],
        )

        base_bench = _benchmark_inference(
            model=model,
            loader=splits.test_loader,
            device=device,
            pinn=cfg.pinn,
            block_diag=cfg.block_diag,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            max_batches=args.bench_max_batches,
        )
        log.info(
            "Base FP32 latency | p50 %.3f ms  p90 %.3f ms  mean %.3f ms",
            base_bench["latency_p50_ms"],
            base_bench["latency_p90_ms"],
            base_bench["latency_mean_ms"],
        )

    # Merge LoRA adapters before quantization (if present)
    merged = merge_lora_weights(model)
    if merged:
        log.info("Merged %d LoRA modules into base weights", len(merged))

    quant_model = _apply_dynamic_quantization(model)
    log.info("Applied dynamic quantization")

    if not args.no_eval:
        quant_eval = evaluate_test(
            model=quant_model,
            test_loader=splits.test_loader,
            device=device,
            pinn=cfg.pinn,
            block_diag=cfg.block_diag,
            show_progress=False,
        )
        quant_metrics = {
            "loss": float(quant_eval.loss),
            "rmse": float(quant_eval.rmse),
            "rmse_mag": float(quant_eval.rmse_mag),
            "rmse_ang_deg": float(quant_eval.rmse_ang_deg),
        }
        log.info(
            "Quantized test | loss %.6g rmse %.6g (mag %.6g, ang %.6g)",
            quant_metrics["loss"],
            quant_metrics["rmse"],
            quant_metrics["rmse_mag"],
            quant_metrics["rmse_ang_deg"],
        )

        quant_bench = _benchmark_inference(
            model=quant_model,
            loader=splits.test_loader,
            device=device,
            pinn=cfg.pinn,
            block_diag=cfg.block_diag,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            max_batches=args.bench_max_batches,
        )
        log.info(
            "Quantized latency | p50 %.3f ms  p90 %.3f ms  mean %.3f ms",
            quant_bench["latency_p50_ms"],
            quant_bench["latency_p90_ms"],
            quant_bench["latency_mean_ms"],
        )

    # Save quantized model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quant_model.state_dict(), out_path)
    log.info("Saved quantized model: %s", out_path)

    base_size = float(os.path.getsize(args.ckpt)) if os.path.isfile(args.ckpt) else 0.0
    quant_size = float(os.path.getsize(out_path)) if out_path.exists() else 0.0
    size_reduction_pct = 100.0 * (1.0 - (quant_size / base_size)) if base_size > 0 else 0.0
    size_reduction_x = (base_size / quant_size) if quant_size > 0 else 0.0

    size_stats = {
        "base_bytes": base_size,
        "quant_bytes": quant_size,
        "size_reduction_pct": size_reduction_pct,
        "size_reduction_x": size_reduction_x,
    }

    speedup_stats: dict[str, float] = {}
    if base_bench and quant_bench:
        base_p50 = float(base_bench["latency_p50_ms"])
        quant_p50 = float(quant_bench["latency_p50_ms"])
        base_p90 = float(base_bench["latency_p90_ms"])
        quant_p90 = float(quant_bench["latency_p90_ms"])

        speedup_stats = {
            "latency_p50_reduction_pct": 100.0 * (1.0 - (quant_p50 / base_p50)) if base_p50 > 0 else 0.0,
            "latency_p90_reduction_pct": 100.0 * (1.0 - (quant_p90 / base_p90)) if base_p90 > 0 else 0.0,
            "latency_p50_speedup_x": (base_p50 / quant_p50) if quant_p50 > 0 else 0.0,
            "latency_p90_speedup_x": (base_p90 / quant_p90) if quant_p90 > 0 else 0.0,
        }

    metric_deltas = _metric_deltas(base_metrics, quant_metrics)

    # Persist summary locally
    summary = {
        "base_metrics": base_metrics,
        "quant_metrics": quant_metrics,
        "metric_deltas": metric_deltas,
        "base_bench": base_bench,
        "quant_bench": quant_bench,
        "size": size_stats,
        "speedup": speedup_stats,
    }
    summary_path = Path(paths.artifacts_dir) / "quant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # CSV summary for quick inspection
    csv_path = Path(paths.artifacts_dir) / "quant_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "base", "quant", "change_pct", "factor_x"])
        for k, base_v in base_metrics.items():
            qv = quant_metrics.get(k, float("nan"))
            pct = metric_deltas.get(f"{k}/change_pct", float("nan"))
            fx = metric_deltas.get(f"{k}/factor_x", float("nan"))
            w.writerow([k, base_v, qv, pct, fx])

    tags = add_basic_tags(repo_root=repo_root)
    tags.update({
        "run_id": run_id,
        "run_slug": run_slug,
        "method": "ptq",
        "quantization": "dynamic_int8",
        "device": str(device),
    })
    if bool(getattr(cfg, "peft", False)):
        tags.update({"peft": "true", "peft_method": str(getattr(cfg, "peft_method", "lora"))})

    experiment = args.mlflow_experiment or cfg.mlflow_experiment

    with mlflow_run(
        enabled=cfg.mlflow,
        strict=bool(getattr(cfg, "mlflow_strict", bool(cfg.mlflow))),
        tracking_uri=cfg.mlflow_tracking_uri,
        experiment=experiment,
        artifact_location=getattr(cfg, "mlflow_artifact_location", None),
        run_name=run_name,
        tags=tags,
    ) as mlf:
        if mlf is None:
            log.warning("MLflow disabled; artifacts stored locally at %s", paths.run_dir)
        else:
            log.info("MLflow enabled; logging metrics + artifacts")
            log_params_safe(mlf, safe_param_dict(cfg))
            mlf.log_param("quant_ckpt", str(args.ckpt))
            mlf.log_param("quant_out", str(out_path))

            try:
                mlf.log_metric("params_total", float(total_params))
            except Exception:
                pass

            for k, v in base_metrics.items():
                try:
                    mlf.log_metric(f"base/test/{k}", float(v))
                except Exception:
                    pass
            for k, v in quant_metrics.items():
                try:
                    mlf.log_metric(f"quant/test/{k}", float(v))
                except Exception:
                    pass
            for k, v in metric_deltas.items():
                try:
                    mlf.log_metric(f"compare/{k}", float(v))
                except Exception:
                    pass

            for k, v in base_bench.items():
                try:
                    mlf.log_metric(f"base/bench/{k}", float(v))
                except Exception:
                    pass
            for k, v in quant_bench.items():
                try:
                    mlf.log_metric(f"quant/bench/{k}", float(v))
                except Exception:
                    pass
            for k, v in speedup_stats.items():
                try:
                    mlf.log_metric(f"compare/{k}", float(v))
                except Exception:
                    pass

            for k, v in size_stats.items():
                try:
                    mlf.log_metric(f"size/{k}", float(v))
                except Exception:
                    pass

            if config_path:
                try:
                    mlf.log_artifact(config_path, artifact_path="config")
                except Exception:
                    pass

        try:
            snapshot_code(
                repo_root=repo_root,
                out_zip=Path(paths.artifacts_dir) / "code_snapshot.zip",
                include_globs=(
                    "train/**/*.py",
                    "models/**/*.py",
                    "data_loading/**/*.py",
                    "scripts/**/*.py",
                    "configs/**/*.yaml",
                    "pyproject.toml",
                    "run_train.py",
                    "README.md",
                ),
            )
        except Exception:
            pass

        if mlf is not None:
            artifact_path = str(getattr(cfg, "mlflow_artifact_path", "run") or "run")
            log_run_artifacts(mlflow=mlf, run_dir=Path(paths.run_dir), artifact_path=artifact_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())