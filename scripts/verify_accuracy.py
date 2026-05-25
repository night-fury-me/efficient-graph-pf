"""Quick verification: train each model on Cora with 3 seeds, report test accuracy.

Usage:
    .venv/bin/python scripts/verify_accuracy.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.examples.ignn_cora import IGNN, _load_cora
from iem.examples.exp_explicit_gnn_extension import (
    ExplicitGCN, ExplicitGIN, ExplicitGAT, ExplicitGraphSAGE, ExplicitAPPNP,
    set_seed, _HP,
)

SEEDS = [42, 137, 271]


def train_and_eval(model_name, model, X, A_hat, y, train_mask, val_mask, test_mask,
                   seed, device):
    set_seed(seed)
    hp = _HP.get(model_name, {"lr": 0.01, "wd": 5e-4, "epochs": 200})
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best_val, best_state = 0.0, None
    patience, patience_counter = 50, 0

    for ep in range(hp["epochs"]):
        model.train()
        logits, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits_v, _ = model(X, A_hat)
            val_acc = float((logits_v.argmax(1)[val_mask] == y[val_mask]).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_t, _ = model(X, A_hat)
        test_acc = float((logits_t.argmax(1)[test_mask] == y[test_mask]).float().mean())
    return test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = _load_cora(Path("datasets/cora"))
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    nf, nc = data["n_features"], data["n_classes"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    models_to_test = [
        ("GCN-2",  lambda: ExplicitGCN(nf, 64, nc, n_layers=2, dropout=0.5).to(device)),
        ("GCN-4",  lambda: ExplicitGCN(nf, 64, nc, n_layers=4, dropout=0.5).to(device)),
        ("GIN-2",  lambda: ExplicitGIN(nf, 128, nc, n_layers=2, dropout=0.5).to(device)),
        ("GAT-2",  lambda: ExplicitGAT(nf, 8, nc, n_layers=2, n_heads=8,
                                        dropout=0.6, attn_dropout=0.6).to(device)),
        ("SAGE-2", lambda: ExplicitGraphSAGE(nf, 64, nc, n_layers=2, dropout=0.5).to(device)),
        ("APPNP",  lambda: ExplicitAPPNP(nf, 64, nc, n_prop=10, alpha=0.1, dropout=0.5).to(device)),
    ]

    print(f"\n{'Model':<10} " + " ".join(f"{'Seed '+str(s):>10}" for s in SEEDS) + f" {'Mean':>10} {'Target':>10}")
    print("-" * 80)

    targets = {"GCN-2": 81, "GCN-4": 79, "GIN-2": 77, "GAT-2": 64, "SAGE-2": 79, "APPNP": 83}

    for model_name, model_fn in models_to_test:
        accs = []
        for seed in SEEDS:
            model = model_fn()
            acc = train_and_eval(model_name, model, X, A_hat, y,
                                 train_mask, val_mask, test_mask, seed, device)
            accs.append(acc)
        mean_acc = np.mean(accs)
        target = targets.get(model_name, "?")
        status = "OK" if mean_acc * 100 >= float(target) - 2 else "LOW"
        print(f"{model_name:<10} " +
              " ".join(f"{a*100:>9.1f}%" for a in accs) +
              f" {mean_acc*100:>9.1f}% {target:>8}%  {status}")


if __name__ == "__main__":
    main()
