from __future__ import annotations

import os

import torch

from .cli import parse_train_config
from .data import build_dataloaders
from .logger import configure_logging, log
from .logging_utils import ensure_run_dirs
from .loop import evaluate_test, train_validate
from .modeling import count_parameters, create_model, init_weights
from .optim_utils import build_optimizer_and_scheduler
from .plotting import plot_history


def main(argv: list[str] | None = None) -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ensure_run_dirs()

    cfg, config_path = parse_train_config(argv)

    log_filename = f"./results/{cfg.runname}_training_log.txt"
    configure_logging(log_file=log_filename)

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
            runname=cfg.runname,
        )
        plot_history(history=history, runname=cfg.runname, pinn=cfg.pinn)

    if "test" in cfg.mode:
        m_test = evaluate_test(
            model=model,
            test_loader=splits.test_loader,
            device=device,
            pinn=cfg.pinn,
            block_diag=cfg.block_diag,
        )
        if cfg.pinn:
            log.info(
                "Test physics-loss : %.4e | total RMSE : %.4e | |V| RMSE : %.4e | θ RMSE : %.4e°",
                m_test.loss,
                m_test.rmse,
                m_test.rmse_mag,
                m_test.rmse_ang_deg,
            )
        else:
            log.info(
                "Final test-set RMSE : %.4e (|V|: %.4e, θ: %.4e°)",
                m_test.rmse,
                m_test.rmse_mag,
                m_test.rmse_ang_deg,
            )

    return 0
