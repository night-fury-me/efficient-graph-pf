"""Cross-voltage zero-shot evaluation for HyperDEQ_PF_Pilot.

Loads a trained HyperDEQ_PF_Pilot from <run_dir>, evaluates on a different
voltage class's dataset (e.g. trained on HVN, evaluated on MVN), and
reports RMSE / mag-RMSE / ang-RMSE / phys_loss on:
  - source-domain test split (sanity check the model still works)
  - target-domain full subset (zero-shot transfer measurement)

Usage:
    python scripts/eval_cross_voltage.py <run_dir> <source_parquet> <target_parquet>

Example:
    python scripts/eval_cross_voltage.py \
        results/runs/260524-XXXX/ \
        ./datasets/HVN_stratified_1500.parquet \
        ./datasets/MVN_stratified_1500.parquet
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import models  # registers all
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from models.registry import build_model
from train.data import build_dataloaders


def _infer_kwargs(run_dir: Path) -> tuple[dict, str]:
    log = (run_dir / "train.log").read_text()
    m = lambda pat, default: (
        type(default)(re.search(pat, log).group(1)) if re.search(pat, log) else default
    )
    kwargs = dict(
        d=m(r"\bd:\s*(\d+)", 4),
        d_hi=m(r"\bd_hi:\s*(\d+)", 32),
        num_attn_layers=m(r"\battn_layers:\s*(\d+)", 2),
        K=m(r"\bK:\s*(\d+)", 15),
        dtheta_max=m(r"DthetaMax:\s*([\d.]+)", 0.30),
        dvm_frac=m(r"DvmFrac:\s*([\d.]+)", 0.10),
        pinn=True,  # force phys readout
    )
    m_name = re.search(r"MODEL:(\S+?),", log)
    model_name = m_name.group(1) if m_name else "HyperDEQ_PF_Pilot"
    return kwargs, model_name


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    parquet: str,
    *,
    batch_size: int = 32,
    split: str = "all",   # "all" = whole subset; or use built-in train/val/test split
    device: torch.device,
) -> dict:
    if split == "all":
        ds = ChanghunDataset([parquet], per_unit=True, device=device)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_blockdiag)
        n_samples_msg = f"all {len(ds)} samples"
    else:
        splits = build_dataloaders(
            parquet_paths=[parquet], per_unit=True, device=device,
            batch_size=batch_size, block_diag=True,
            seed=42, split_mode="ratio", train_ratio=0.8, valid_ratio=0.1,
        )
        loader = {"train": splits.train_loader, "val": splits.val_loader, "test": splits.test_loader}[split]
        n_samples_msg = f"{split} split"

    sum_se = sum_mag = sum_ang = sum_phys = 0.0
    n = 0
    for batch in loader:
        bus_type = batch["bus_type"].to(device)
        Line = batch["Lines_connected"].to(device)
        Ys = batch["Y_Lines"].to(device)
        Yc = batch["Y_C_Lines"].to(device)
        S = batch["S_start"].to(device)
        V0 = batch["V_start"].to(device)
        Vt = batch["V_newton"].to(device)
        sizes = batch["sizes"].to(device)

        out = model(bus_type, Line, None, Ys, Yc, S, V0, sizes)
        V_pred, phys = (out if isinstance(out, tuple) else (out, None))

        B = bus_type.size(0)
        diff = V_pred - Vt
        sum_se += float((diff * diff).mean().item()) * B
        sum_mag += float((diff[..., 0] ** 2).mean().item()) * B
        sum_ang += float((diff[..., 1] ** 2).mean().item()) * B
        if phys is not None:
            sum_phys += float(phys.item()) * B
        n += B

    return dict(
        n_batches=n,
        rmse=(sum_se / n) ** 0.5,
        rmse_mag=(sum_mag / n) ** 0.5,
        rmse_ang_deg=(sum_ang / n) ** 0.5 * 180.0 / math.pi,
        phys_loss=sum_phys / n,
        eval_set=n_samples_msg,
    )


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: python scripts/eval_cross_voltage.py <run_dir> <source_parquet> <target_parquet>",
              file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    source_pq = sys.argv[2]
    target_pq = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs, model_name = _infer_kwargs(run_dir)
    print(f"Model: {model_name}")
    print(f"Kwargs: {kwargs}")
    print(f"Source domain: {source_pq}")
    print(f"Target domain: {target_pq}")
    print()

    model = build_model(model_name, device=device, **kwargs)
    state = torch.load(run_dir / "ckpt" / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("=== Source domain (test split, sanity check) ===")
    source_res = evaluate(model, source_pq, split="test", device=device)
    print(json.dumps(source_res, indent=2))
    print()

    print("=== Target domain (all samples, zero-shot transfer) ===")
    target_res = evaluate(model, target_pq, split="all", device=device)
    print(json.dumps(target_res, indent=2))
    print()

    print("=== Cross-voltage transfer ratio ===")
    ratio = target_res["rmse"] / source_res["rmse"]
    print(f"  target.rmse / source.rmse = {ratio:.2f}")
    print(f"  go condition: ratio < 5.0 -> {'PASS' if ratio < 5.0 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
