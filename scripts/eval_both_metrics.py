"""Post-hoc evaluator: compute val_rmse AND val_phys_loss for any
trained checkpoint, regardless of how the model was trained.

Why: history.csv only logs the training-loss metric (phys when pinn=True,
MSE when pinn=False). To compare models on BOTH metrics simultaneously,
we re-evaluate the best checkpoint on the val loader, forcing the model
to also report its KCL residual.

Usage:
    python -m scripts.eval_both_metrics <run_dir> <model_name>
    e.g.  python -m scripts.eval_both_metrics results/runs/260522-204937_7968/ GNSMsg_EdgeSelfAttn
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import models  # noqa: F401 -- registers all model names
from data_loading.collate import collate_blockdiag  # noqa: F401
from models.registry import build_model
from train.data import build_dataloaders


def _infer_model_kwargs(run_dir: Path) -> tuple[dict, str]:
    """Read the train.log to recover the architecture kwargs."""
    log = (run_dir / "train.log").read_text()
    # CLI args dump is at the top of the log
    m = lambda pat, default: (
        type(default)(re.search(pat, log).group(1)) if re.search(pat, log) else default
    )
    # The train log dumps "MODEL:..., d:4, d_hi:32, attn_layers:2, K:15, ..."
    # NOTE: the field is `attn_layers:N`, not `num_attn_layers:N`.
    kwargs = dict(
        d=m(r"\bd:\s*(\d+)", 4),
        d_hi=m(r"\bd_hi:\s*(\d+)", 16),
        num_attn_layers=m(r"\battn_layers:\s*(\d+)", 1),
        K=m(r"\bK:\s*(\d+)", 10),
        dtheta_max=m(r"DthetaMax:\s*([\d.]+)", 0.30),
        dvm_frac=m(r"DvmFrac:\s*([\d.]+)", 0.10),
        gamma=0.9,
        v_limit=True,
        use_armijo=True,
        pinn=True,
    )
    slug = json.loads((run_dir / "meta.json").read_text())["run_slug"]
    return kwargs, slug


@torch.no_grad()
def evaluate(run_dir: Path, model_name: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs, slug = _infer_model_kwargs(run_dir)

    model = build_model(model_name, device=device, **kwargs)
    state = torch.load(run_dir / "ckpt" / "best.ckpt", map_location=device, weights_only=False)
    # strict=True now that architecture inference is fixed -- any mismatch
    # should be a loud failure, not a silent drop of weights.
    model.load_state_dict(state, strict=True)
    # Force pinn=True so the model returns the KCL residual.
    if hasattr(model, "pinn"):
        model.pinn = True
    model.eval()

    splits = build_dataloaders(
        parquet_paths=["./datasets/HVN_15000_NR_plain_4_to_32_buses.parquet"],
        per_unit=True,
        device=device,
        batch_size=64,
        block_diag=True,
        seed=42,
        split_mode="ratio",
        train_ratio=0.8,
        valid_ratio=0.1,
    )

    sum_phys = 0.0
    sum_se = 0.0
    sum_se_mag = 0.0
    sum_se_ang = 0.0
    n = 0
    for batch in splits.val_loader:
        n_nodes = batch["sizes"].to(device)
        bus_type = batch["bus_type"].to(device)
        Line = batch["Lines_connected"].to(device)
        Ys = batch["Y_Lines"].to(device)
        Yc = batch["Y_C_Lines"].to(device)
        S = batch["S_start"].to(device)
        V0 = batch["V_start"].to(device)
        Vt = batch["V_newton"].to(device)

        out = model(bus_type, Line, None, Ys, Yc, S, V0, n_nodes)
        if isinstance(out, tuple):
            V_pred, phys = out
        else:
            V_pred, phys = out, None

        B = bus_type.size(0)
        if phys is not None:
            sum_phys += float(phys.item()) * B

        diff = V_pred - Vt
        se = (diff * diff).mean()
        # Angle wrap then split
        v_diff = diff[..., 0]
        a_diff = diff[..., 1]
        # angle is already wrapped; just MSE
        se_mag = (v_diff * v_diff).mean()
        se_ang = (a_diff * a_diff).mean()
        sum_se += float(se.item()) * B
        sum_se_mag += float(se_mag.item()) * B
        sum_se_ang += float(se_ang.item()) * B
        n += B

    return dict(
        slug=slug,
        n=n,
        val_phys_loss=sum_phys / n,
        val_rmse=(sum_se / n) ** 0.5,
        val_rmse_mag=(sum_se_mag / n) ** 0.5,
        val_rmse_ang_deg=((sum_se_ang / n) ** 0.5) * 180.0 / 3.141592653589793,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m scripts.eval_both_metrics <run_dir> <model_name>", file=sys.stderr)
        sys.exit(1)
    run_dir = Path(sys.argv[1]).resolve()
    model_name = sys.argv[2]
    result = evaluate(run_dir, model_name)
    print(json.dumps(result, indent=2))
