from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from models.gnsmsg_armijo import GNSMsg
from models.gnsmsg_edge_selfattn_armijo import GNSMsg_EdgeSelfAttn


def create_model(
    *,
    model_name: str,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    num_attn_layers: int,
    device: torch.device,
) -> nn.Module:
    if model_name == "GNSMsg":
        model = GNSMsg(
            d=d,
            d_hi=d_hi,
            K=K,
            pinn=pinn,
            gamma=gamma,
            v_limit=v_limit,
            use_armijo=use_armijo,
        ).to(device)
    elif model_name == "GNSMsg_EdgeSelfAttn":
        model = GNSMsg_EdgeSelfAttn(
            d=d,
            d_hi=d_hi,
            K=K,
            pinn=pinn,
            gamma=gamma,
            v_limit=v_limit,
            use_armijo=use_armijo,
            num_attn_layers=num_attn_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def init_weights(model: nn.Module, *, weight_init: str, bias_init: float, exclude_modules: List[nn.Module] | None = None) -> None:
    exclude_modules = exclude_modules or []

    for module in model.modules():
        if module in exclude_modules:
            continue

        if isinstance(module, nn.Linear):
            if weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                module.bias.data.fill_(bias_init)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

        else:
            for name, param in module.named_parameters(recurse=False):
                if "weight" in name and param.dim() > 1:
                    if weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity="relu")
                elif "bias" in name:
                    param.data.fill_(bias_init)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
