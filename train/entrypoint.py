from __future__ import annotations

import os
import tempfile

import torch

from .cli import parse_train_config
from .data import build_dataloaders
from .logger import configure_logging, log
from .loop import evaluate_test, train_validate
from .modeling import count_parameters, create_model, init_weights
from .optim_utils import build_optimizer_and_scheduler
from pathlib import Path

from .mlflow_utils import add_basic_tags, log_params_safe, mlflow_run, snapshot_code
from .run_naming import make_run_id, make_run_slug, safe_param_dict


def main(argv: list[str] | None = None) -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cfg, config_path = parse_train_config(argv)

    # When MLflow is enabled, we avoid persisting local run folders and instead
    # stage artifacts in a temporary directory and log them to MLflow.
    configure_logging(log_file=None)

    if config_path:
        log.info("Loaded config: %s", config_path)

    log.info(
        "MODEL:%s, PINN:%s, Block:%s, d:%s, d_hi:%s,K:%s, Runname:%s, PARQUET:%s, BATCH:%s, EP:%s, LR:%s",
        cfg.model_name,
        cfg.pinn,
        cfg.block_diag,
        cfg.d,
        cfg.d_hi,
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

    repo_root = Path(__file__).resolve().parents[1]
    tags = add_basic_tags(repo_root=repo_root)
    tags.update({"device": str(device)})

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
    tags.update({"run_id": run_id, "run_slug": run_slug})

    with mlflow_run(
        enabled=cfg.mlflow,
        tracking_uri=cfg.mlflow_tracking_uri,
        experiment=cfg.mlflow_experiment,
        run_name=run_name,
        tags=tags,
    ) as mlf:
        if mlf is None:
            log.warning("MLflow disabled/unavailable; no MLflow logging will occur.")
        else:
            # Log hparams (no dataset contents; only config values/paths).
            log_params_safe(mlf, safe_param_dict(cfg))

            # Log config file used (artifact) if provided.
            if config_path:
                try:
                    mlf.log_artifact(config_path, artifact_path="config")
                except Exception:
                    pass

            # Snapshot code used for this run as a single artifact (zip).
            with tempfile.TemporaryDirectory(prefix="mlflow_stage_") as td:
                stage = Path(td)

                code_zip = stage / "code_snapshot.zip"
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
                try:
                    mlf.log_artifact(str(code_zip), artifact_path="code")
                except Exception:
                    pass

                # Collect full history to log as artifact (CSV) for plotting.
                rows: dict[tuple[int, str], dict[str, float]] = {}

                def on_epoch_metrics(epoch: int, split: str, m):
                    # Split is 'train' or 'val'
                    metrics = {
                        f"{split}/loss": float(m.loss),
                        f"{split}/rmse": float(m.rmse),
                        f"{split}/rmse_mag": float(m.rmse_mag),
                        f"{split}/rmse_ang_deg": float(m.rmse_ang_deg),
                    }
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

                best_ckpt_path = stage / "best.ckpt"
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
                        best_ckpt_path=str(best_ckpt_path),
                        on_epoch_metrics=on_epoch_metrics,
                    )

                    # Log best checkpoint only (selected by val rmse_mag+rmse_ang_deg).
                    try:
                        mlf.log_artifact(str(best_ckpt_path), artifact_path="ckpt")
                    except Exception:
                        pass

                    # Record which epoch was best + best metrics.
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
                    try:
                        mlf.log_metric("test/loss", float(m_test.loss))
                        mlf.log_metric("test/rmse", float(m_test.rmse))
                        mlf.log_metric("test/rmse_mag", float(m_test.rmse_mag))
                        mlf.log_metric("test/rmse_ang_deg", float(m_test.rmse_ang_deg))
                    except Exception:
                        pass

                # Log history table as CSV artifact for easy plotting.
                try:
                    import csv

                    hist_csv = stage / "history.csv"
                    with hist_csv.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "rmse", "rmse_mag", "rmse_ang_deg"])
                        w.writeheader()
                        for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                            w.writerow({
                                "epoch": int(row["epoch"]),
                                "split": row["split"],
                                "loss": row["loss"],
                                "rmse": row["rmse"],
                                "rmse_mag": row["rmse_mag"],
                                "rmse_ang_deg": row["rmse_ang_deg"],
                            })
                    mlf.log_artifact(str(hist_csv), artifact_path="metrics")
                except Exception:
                    pass

                # Optionally generate plots from history and log them.
                if history is not None:
                    try:
                        from .plotting import plot_history

                        plots_dir = stage / "plots"
                        plot_history(history=history, pinn=cfg.pinn, plots_dir=str(plots_dir))
                        mlf.log_artifacts(str(plots_dir), artifact_path="plots")
                    except Exception:
                        pass

    return 0
