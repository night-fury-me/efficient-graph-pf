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
        init_weights(model, weight_init=cfg.weight_init, bias_init=cfg.bias_init, exclude_modules=[])
        log.info("Total number of parameters: %s", count_parameters(model))

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

            if "train" in cfg.mode:
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
