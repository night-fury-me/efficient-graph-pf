from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from data_loading.samplers import MultiBucketBatchSampler


@dataclass(frozen=True)
class DataSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_train: int
    n_val: int
    n_test: int


def build_dataloaders(
    *,
    parquet_paths,
    per_unit: bool,
    device: torch.device,
    batch_size: int,
    block_diag: bool,
    seed: int,
    split_mode: str,
    train_ratio: float,
    valid_ratio: float,
) -> DataSplits:
    full_ds = ChanghunDataset(parquet_paths, per_unit=per_unit, device=device)

    n_total = len(full_ds)
    if split_mode == "equal3":
        base, rem = divmod(n_total, 3)
        # Distribute remainder to train then valid (difference between splits <= 1)
        n_train = base + (1 if rem > 0 else 0)
        n_val = base + (1 if rem > 1 else 0)
        n_test = n_total - n_train - n_val
    else:
        n_train = int(train_ratio * n_total)
        n_val = int(valid_ratio * n_total)
        n_test = n_total - n_train - n_val

    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError(
            f"Invalid split sizes: n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}. "
            f"Check split_mode/train_ratio/valid_ratio."
        )

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    if batch_size == 1:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        if block_diag:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_blockdiag
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_blockdiag
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_blockdiag
            )
        else:
            sizes = [full_ds[i]["N"] for i in range(len(full_ds))]
            train_sampler = MultiBucketBatchSampler(
                sizes=np.take(sizes, train_ds.indices),
                batch_size=batch_size,
                shuffle=True,
            )
            val_sampler = MultiBucketBatchSampler(
                sizes=np.take(sizes, val_ds.indices),
                batch_size=batch_size,
                shuffle=False,
            )
            test_sampler = MultiBucketBatchSampler(
                sizes=np.take(sizes, test_ds.indices),
                batch_size=batch_size,
                shuffle=False,
            )
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
            test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

    return DataSplits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
