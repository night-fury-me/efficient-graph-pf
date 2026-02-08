from __future__ import annotations

import os
import json
import csv
import tempfile
from contextlib import ExitStack

import torch

from .cli import parse_train_config
from .data import build_dataloaders
from .logger import configure_logging, log
from .loop import evaluate_test, train_validate
from .modeling import count_parameters, create_model, init_weights
from .optim_utils import build_optimizer_and_scheduler
from pathlib import Path

from .mlflow_utils import add_basic_tags, log_params_safe, log_run_artifacts, mlflow_run, snapshot_code
from .run_naming import make_run_id, make_run_slug, safe_param_dict
from .logging_utils import ensure_run_dirs, make_run_paths
from .peft_utils import (
    apply_lora_to_linear_modules,
    count_trainable_params,
    freeze_all,
    freeze_all_except_lora,
    unfreeze_modules,
)


def main(argv: list[str] | None = None) -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cfg, config_path = parse_train_config(argv)

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
    run_id = make_run_id(run_slug=run_slug)
    run_name = f"{run_id}_{run_slug}"

    # Local persistence strategy:
    # - MLflow disabled OR keep_local_run_dir=true: persist under results/runs/<run_id>
    # - MLflow enabled and keep_local_run_dir=false: stage in a temp dir, upload to MLflow, then delete
    keep_local = (not bool(cfg.mlflow)) or bool(getattr(cfg, "mlflow_keep_local_run_dir", False))

    with ExitStack() as stack:
        if keep_local:
            paths = make_run_paths(run_id=run_id, base_dir="./results/runs")
        else:
            staging_root = Path("./results/.mlflow_staging")
            staging_root.mkdir(parents=True, exist_ok=True)
            tmp_base = stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f"{run_id}_", dir=str(staging_root))
            )
            paths = make_run_paths(run_id=run_id, base_dir=tmp_base)

        ensure_run_dirs(paths)
        log_file = str(Path(paths.run_dir) / "train.log")
        configure_logging(log_file=log_file)

        if config_path:
            log.info("Loaded config: %s", config_path)

        log.info(
            "MODEL:%s, PINN:%s, Block:%s, d:%s, d_hi:%s, attn_layers:%s, K:%s, Runname:%s, PARQUET:%s, BATCH:%s, EP:%s, LR:%s",
            cfg.model_name,
            cfg.pinn,
            cfg.block_diag,
            cfg.d,
            cfg.d_hi,
            cfg.num_attn_layers,
            cfg.K,
            cfg.runname,
            cfg.parquet_paths,
            cfg.batch_size,
            cfg.epochs,
            cfg.lr,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            train_subset_frac=getattr(cfg, "train_subset_frac", None),
            train_subset_min_n=int(getattr(cfg, "train_subset_min_n", 1)),
        )
        log.info(
            "Dataset sizes | train %s  valid %s  test %s",
            splits.n_train,
            splits.n_val,
            splits.n_test,
        )

        model = create_model(
            model_name=cfg.model_name,
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

        # Optional: initialize from a pretrained/base checkpoint before training.
        # - Full fine-tuning: set run.init_ckpt_path
        # - PEFT/LoRA: either set run.init_ckpt_path or peft.base_ckpt_path
        ckpt_path = None
        if getattr(cfg, "init_ckpt_path", None):
            ckpt_path = str(getattr(cfg, "init_ckpt_path"))
        elif bool(getattr(cfg, "peft", False)) and getattr(cfg, "peft_base_ckpt_path", None):
            ckpt_path = str(getattr(cfg, "peft_base_ckpt_path"))

        if ckpt_path:
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=True)
                log.info("Loaded checkpoint: %s", ckpt_path)
            except Exception as e:
                log.exception("Failed to load checkpoint '%s': %s", ckpt_path, e)
                raise
        else:
            init_weights(model, weight_init=cfg.weight_init, bias_init=cfg.bias_init, exclude_modules=[])

        # Optional: apply PEFT (LoRA)
        if bool(getattr(cfg, "peft", False)):
            method = str(getattr(cfg, "peft_method", "lora")).lower().strip()
            if method != "lora":
                raise ValueError(f"Unsupported peft_method: {method}. Supported: lora")

            wrapped = apply_lora_to_linear_modules(
                model,
                target_module_names=list(getattr(cfg, "lora_target_modules", ["q", "k", "v", "out"])),
                r=int(getattr(cfg, "lora_r", 8)),
                alpha=int(getattr(cfg, "lora_alpha", 16)),
                dropout=float(getattr(cfg, "lora_dropout", 0.0)),
            )
            log.info("Applied LoRA to %d Linear modules", len(wrapped))
            if wrapped:
                log.info("LoRA wrapped modules (first 12): %s", wrapped[:12])

            train_base = bool(getattr(cfg, "peft_train_base", False))
            if not train_base:
                freeze_all_except_lora(model)

                extra = list(getattr(cfg, "peft_unfreeze_modules", []) or [])
                if extra:
                    unfrozen = unfreeze_modules(model, extra)
                    if unfrozen:
                        log.info("PEFT: additionally unfroze modules: %s", unfrozen)
                    else:
                        log.warning(
                            "PEFT: peft_unfreeze_modules was set but no modules were unfrozen: %s",
                            extra,
                        )

        if bool(getattr(cfg, "peft", False)) and bool(getattr(cfg, "head_only_ft", False)):
            log.warning("Head-only FT is enabled with PEFT; head-only freezing will override PEFT freezing.")

        # Optional: head-only fine-tuning (no LoRA). Freeze all then unfreeze heads.
        if bool(getattr(cfg, "head_only_ft", False)):
            freeze_all(model)
            head_modules = list(getattr(cfg, "head_only_modules", []) or [])
            if head_modules:
                unfrozen = unfreeze_modules(model, head_modules)
                if unfrozen:
                    log.info("Head-only FT: unfroze modules: %s", unfrozen)
                else:
                    log.warning(
                        "Head-only FT: head_only_modules was set but no modules were unfrozen: %s",
                        head_modules,
                    )
            else:
                log.warning("Head-only FT enabled but no head_only_modules provided; all params frozen.")

        total_params = int(count_parameters(model))
        trainable_params = int(count_trainable_params(model))
        trainable_frac = (float(trainable_params) / float(total_params)) if total_params > 0 else 0.0
        trainable_pct = 100.0 * trainable_frac
        reduction_pct = 100.0 * (1.0 - trainable_frac)
        reduction_x = (float(total_params) / float(trainable_params)) if trainable_params > 0 else float("inf")

        log.info("Total parameters: %s", total_params)
        log.info("Trainable parameters: %s", trainable_params)
        log.info(
            "Parameter efficiency: trainable %.2f%% | reduced %.2f%% | %.2fx fewer trainable params",
            trainable_pct,
            reduction_pct,
            reduction_x,
        )

        param_eff = {
            "params_total": float(total_params),
            "params_trainable": float(trainable_params),
            "params_trainable_pct": float(trainable_pct),
            "params_reduction_pct": float(reduction_pct),
            "params_reduction_x": float(reduction_x),
        }

        optim_bundle = None
        if "train" in cfg.mode:
            optim_bundle = build_optimizer_and_scheduler(
                model=model,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                lr_scheduler=cfg.lr_scheduler,
                cosine_restart_epoch=cfg.cosine_restart_epoch,
                steps_per_epoch=len(splits.train_loader),
            )

        tags = add_basic_tags(repo_root=repo_root)
        tags.update({"device": str(device)})
        tags.update({"run_id": run_id, "run_slug": run_slug})
        tags.update({"seed": str(cfg.seed)})
        if getattr(cfg, "train_subset_frac", None) is not None:
            tags.update({"target_budget": str(getattr(cfg, "train_subset_frac"))})
        if config_path:
            tags.update({"config": str(config_path)})

        # PEFT tags
        if bool(getattr(cfg, "peft", False)):
            tags.update({
                "peft": "true",
                "peft_method": str(getattr(cfg, "peft_method", "lora")),
                "lora_r": str(getattr(cfg, "lora_r", "")),
                "lora_alpha": str(getattr(cfg, "lora_alpha", "")),
                "lora_dropout": str(getattr(cfg, "lora_dropout", "")),
                "peft_train_base": str(bool(getattr(cfg, "peft_train_base", False))).lower(),
                "peft_unfreeze_modules": ",".join(list(getattr(cfg, "peft_unfreeze_modules", []) or [])),
            })

        if bool(getattr(cfg, "head_only_ft", False)):
            tags.update({
                "head_only_ft": "true",
                "head_only_modules": ",".join(list(getattr(cfg, "head_only_modules", []) or [])),
            })

        # Always tag param efficiency for filtering/aggregation.
        tags.update(
            {
                "params_total": str(total_params),
                "params_trainable": str(trainable_params),
                "params_trainable_pct": f"{trainable_pct:.4f}",
                "params_reduction_pct": f"{reduction_pct:.4f}",
                "params_reduction_x": f"{reduction_x:.6f}",
            }
        )

        # Persist basic run metadata locally (staged if MLflow-only mode).
        try:
            meta = {
                "run_id": run_id,
                "run_slug": run_slug,
                "run_name": run_name,
                "config_path": config_path,
                "log_file": log_file,
                "mlflow_enabled": bool(cfg.mlflow),
                "keep_local_run_dir": bool(keep_local),
            }
            (Path(paths.run_dir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

        with mlflow_run(
            enabled=cfg.mlflow,
            strict=bool(getattr(cfg, "mlflow_strict", bool(cfg.mlflow))),
            tracking_uri=cfg.mlflow_tracking_uri,
            experiment=cfg.mlflow_experiment,
            artifact_location=getattr(cfg, "mlflow_artifact_location", None),
            run_name=run_name,
            tags=tags,
        ) as mlf:
            if mlf is None:
                log.info("MLflow disabled; writing artifacts to %s", paths.run_dir)
            else:
                log.info("MLflow enabled; logging metrics + artifacts to MLflow")

                # Log hparams (no dataset contents; only config values/paths).
                log_params_safe(mlf, safe_param_dict(cfg))

                # Log parameter efficiency metrics (single-value, step-less).
                try:
                    for k, v in param_eff.items():
                        mlf.log_metric(k, float(v))
                except Exception:
                    pass

                # Log target budget (few-shot) as a metric as well for easy charting.
                if getattr(cfg, "train_subset_frac", None) is not None:
                    try:
                        mlf.log_metric("target_budget", float(getattr(cfg, "train_subset_frac")))
                    except Exception:
                        pass

                # Log config file used (artifact) if provided.
                if config_path:
                    try:
                        mlf.log_artifact(config_path, artifact_path="config")
                    except Exception:
                        pass

            # Snapshot code used for this run as a single artifact (zip) into the run folder.
            code_zip = Path(paths.artifacts_dir) / "code_snapshot.zip"
            try:
                snapshot_code(
                    repo_root=repo_root,
                    out_zip=code_zip,
                    include_globs=(
                        "train/**/*.py",
                        "models/**/*.py",
                        "data_loading/**/*.py",
                        "configs/**/*.yaml",
                        "pyproject.toml",
                        "run_train.py",
                        "README.md",
                    ),
                )
            except Exception:
                pass

            # Collect full history to log as artifact (CSV) for plotting.
            rows: dict[tuple[int, str], dict[str, float]] = {}

            def on_epoch_metrics(epoch: int, split: str, m):
                metrics = {
                    f"{split}/loss": float(m.loss),
                    f"{split}/rmse": float(m.rmse),
                    f"{split}/rmse_mag": float(m.rmse_mag),
                    f"{split}/rmse_ang_deg": float(m.rmse_ang_deg),
                }
                if mlf is not None:
                    for k, v in metrics.items():
                        try:
                            mlf.log_metric(k, v, step=int(epoch))
                        except Exception:
                            pass
                rows[(int(epoch), split)] = {
                    "epoch": float(epoch),
                    "split": split,
                    "loss": float(m.loss),
                    "rmse": float(m.rmse),
                    "rmse_mag": float(m.rmse_mag),
                    "rmse_ang_deg": float(m.rmse_ang_deg),
                }

            best_ckpt_path = str(Path(paths.ckpt_dir) / "best.ckpt")
            history = None

            final_metrics: dict[str, float] = {}

            if "train" in cfg.mode:
                assert optim_bundle is not None
                history = train_validate(
                    model=model,
                    train_loader=splits.train_loader,
                    val_loader=splits.val_loader,
                    device=device,
                    pinn=cfg.pinn,
                    block_diag=cfg.block_diag,
                    optim=optim_bundle.optim,
                    scheduler=optim_bundle.scheduler,
                    epochs=cfg.epochs,
                    val_every=cfg.val_every,
                    best_ckpt_path=best_ckpt_path,
                    on_epoch_metrics=on_epoch_metrics,
                )

                if mlf is not None:
                    try:
                        mlf.log_metric("best/epoch", float(history.best_epoch))
                        mlf.log_metric("best/score", float(history.best_score))
                        mlf.log_metric("best/val_rmse_mag", float(history.best_val_rmse_mag))
                        mlf.log_metric("best/val_rmse_ang_deg", float(history.best_val_rmse_ang_deg))
                    except Exception:
                        pass

                final_metrics["best/epoch"] = float(history.best_epoch)
                final_metrics["best/score"] = float(history.best_score)
                final_metrics["best/val_rmse_mag"] = float(history.best_val_rmse_mag)
                final_metrics["best/val_rmse_ang_deg"] = float(history.best_val_rmse_ang_deg)

            if "test" in cfg.mode:
                m_test = evaluate_test(
                    model=model,
                    test_loader=splits.test_loader,
                    device=device,
                    pinn=cfg.pinn,
                    block_diag=cfg.block_diag,
                )
                if mlf is not None:
                    try:
                        mlf.log_metric("test/loss", float(m_test.loss))
                        mlf.log_metric("test/rmse", float(m_test.rmse))
                        mlf.log_metric("test/rmse_mag", float(m_test.rmse_mag))
                        mlf.log_metric("test/rmse_ang_deg", float(m_test.rmse_ang_deg))
                    except Exception:
                        pass

                final_metrics["test/loss"] = float(m_test.loss)
                final_metrics["test/rmse"] = float(m_test.rmse)
                final_metrics["test/rmse_mag"] = float(m_test.rmse_mag)
                final_metrics["test/rmse_ang_deg"] = float(m_test.rmse_ang_deg)

            # Optional: compare metrics vs a baseline MLflow run.
            if (
                mlf is not None
                and bool(getattr(cfg, "compare", False))
                and getattr(cfg, "compare_baseline_run_id", None)
            ):
                baseline_run_id = str(getattr(cfg, "compare_baseline_run_id"))
                metric_keys = list(getattr(cfg, "compare_metrics", []))

                try:
                    from mlflow.tracking import MlflowClient  # type: ignore

                    client = MlflowClient()
                    base_run = client.get_run(baseline_run_id)
                    base_metrics = dict(getattr(base_run.data, "metrics", {}) or {})

                    log.info("Comparing metrics vs baseline MLflow run_id=%s", baseline_run_id)
                    try:
                        mlf.set_tag("compare_baseline_run_id", baseline_run_id)
                    except Exception:
                        pass

                    for key in metric_keys:
                        if key not in final_metrics:
                            continue
                        if key not in base_metrics:
                            continue

                        cur = float(final_metrics[key])
                        base = float(base_metrics[key])
                        if base == 0.0 or not (base == base) or not (cur == cur):
                            continue

                        # Coherent convention:
                        # - For RMSE-like metrics, LOWER is better.
                        # - We report signed percent change: (cur-base)/base * 100.
                        #     negative => improved (reduced RMSE)
                        #     positive => worse (increased RMSE)
                        # - We also report a signed factor (x):
                        #     negative magnitude => how many times LOWER (base/cur)
                        #     positive magnitude => how many times HIGHER (cur/base)
                        # Example: -3.36x means 3.36x reduction in RMSE.

                        if base == 0.0:
                            continue

                        pct_change = 100.0 * ((cur - base) / base)
                        if cur == 0.0:
                            factor_x = float("-inf") if pct_change < 0 else float("inf")
                        else:
                            factor_x = -abs(base / cur) if cur < base else abs(cur / base)

                        log.info(
                            "Compare %s | base=%.6g cur=%.6g | rmse_change=%+.2f%% | rmse_factor=%+.3fx",
                            key,
                            base,
                            cur,
                            pct_change,
                            factor_x,
                        )

                        safe_key = str(key).replace("/", "_")
                        try:
                            mlf.log_metric(f"compare/{safe_key}/base", base)
                            mlf.log_metric(f"compare/{safe_key}/cur", cur)
                            mlf.log_metric(f"compare/{safe_key}/rmse_change_pct", pct_change)
                            mlf.log_metric(f"compare/{safe_key}/rmse_factor_x", factor_x)
                        except Exception:
                            pass

                except Exception as e:
                    log.warning("Baseline comparison skipped (%s)", e)

            try:
                hist_csv = Path(paths.artifacts_dir) / "history.csv"
                with hist_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=["epoch", "split", "loss", "rmse", "rmse_mag", "rmse_ang_deg"],
                    )
                    w.writeheader()
                    for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                        w.writerow(
                            {
                                "epoch": int(row["epoch"]),
                                "split": row["split"],
                                "loss": row["loss"],
                                "rmse": row["rmse"],
                                "rmse_mag": row["rmse_mag"],
                                "rmse_ang_deg": row["rmse_ang_deg"],
                            }
                        )
            except Exception:
                pass

            if history is not None:
                try:
                    from .plotting import plot_history

                    plot_history(history=history, pinn=cfg.pinn, plots_dir=str(paths.plots_dir))
                except Exception:
                    pass

            if mlf is not None:
                artifact_path = str(getattr(cfg, "mlflow_artifact_path", "run") or "run")
                log_run_artifacts(mlflow=mlf, run_dir=Path(paths.run_dir), artifact_path=artifact_path)

        return 0
