"""Shared loader + training helpers for revision R2 scripts.

The actual `iem.examples.ignn_*` loaders return a dict (X, A_hat, y, masks,
n_features, n_classes). This helper unwraps them into a uniform tuple and
provides a consistent train_ignn() so each script doesn't have to repeat
the same boilerplate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora
from iem.examples.ignn_citeseer_pubmed import _download_planetoid, _load_planetoid
from iem.examples.ignn_wikics import _download_wikics, _load_wikics
from iem.examples.ignn_amazon import _download_amazon, _load_amazon

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
DATA_ROOT = PROJ_ROOT / "datasets"


def _ensure(name: str):
    d = DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


LOADERS = {
    "Cora":     lambda: _load_cora(_ensure("cora")),
    "Citeseer": lambda: _load_planetoid("citeseer", _ensure("citeseer")),
    "Pubmed":   lambda: _load_planetoid("pubmed", _ensure("pubmed")),
    "WikiCS":   lambda: _load_wikics(_ensure("wikics")),
    "Amazon":   lambda: _load_amazon(_ensure("amazon_photo")),
}

DOWNLOADERS = {
    "Cora":     lambda: _download_cora(_ensure("cora")),
    "Citeseer": lambda: _download_planetoid("citeseer", _ensure("citeseer")),
    "Pubmed":   lambda: _download_planetoid("pubmed", _ensure("pubmed")),
    "WikiCS":   lambda: _download_wikics(_ensure("wikics")),
    "Amazon":   lambda: _download_amazon(_ensure("amazon_photo")),
}


def load_dataset(name: str):
    """Return (X, A_hat, y, train_mask, n_features, n_classes) for the named dataset.

    Downloads if files missing.
    """
    try:
        d = LOADERS[name]()
    except FileNotFoundError:
        DOWNLOADERS[name]()
        d = LOADERS[name]()
    return (d["X"], d["A_hat"], d["y"], d["train_mask"],
            int(d["n_features"]), int(d["n_classes"]))


def train_ignn(X, A_hat, y, train_mask, n_features, n_classes,
               device, seed, epochs: int = 400, hidden: int = 64,
               c: float | None = 0.9, dropout: float = 0.5,
               lr: float = 0.01, wd: float = 5e-4,
               fwd_iter: int = 100, fwd_tol: float = 1e-6,
               cosine: bool = True):
    """Train an IGNN classifier and return the model in eval() mode.

    Uses the validated AEGIS revision-R2 recipe by default: a hard spectral cap
    ``c=0.9`` (genuinely contractive, kappa=||J_z||_2 < 1), dropout 0.5 (train
    only, on Z* before the head), a moderately tight forward solve
    (max_iter=100, tol=1e-6), cosine LR over 400 epochs, Adam lr=0.01 wd=5e-4,
    hidden=64. Reproduces Cora ~80% / Citeseer ~69-70% / Pubmed ~79%.

    Signature is backward compatible: the original positional args plus the
    optional ``epochs`` / ``hidden`` still work; the recipe knobs are extra
    optional kwargs (pass ``c=None`` to recover the legacy ||W||=1 model).
    """
    torch.manual_seed(seed)
    model = IGNN(n_features, hidden=hidden, n_classes=n_classes,
                 c=c, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    for _ in range(epochs):
        model.train()
        logits, _, _ = model(X, A_hat, max_iter=fwd_iter, tol=fwd_tol,
                             train_dropout=(dropout > 0))
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            sched.step()
    model.eval()
    return model


def reconverge(model, Z_init, ctx_sub, max_iter: int = 200, tol: float = 1e-7):
    """Re-iterate the IGNN operator on a subgraph until fixed point."""
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < tol:
                Z = Z_new
                break
            Z = Z_new
    return Z


def forward_and_subgraph(model, X, A_hat, max_nodes: int = 50):
    """Full forward + BFS subgraph + reconverge.

    Returns: (X_sub, A_sub, Z_sub, ctx_sub, full_ctx, full_Z, idx).
    ctx_sub matches what model.operator expects on the subgraph.
    """
    from iem.adversarial import extract_ego_subgraph
    with torch.no_grad():
        _, Z_full, ctx_full = model(X, A_hat)
    idx = extract_ego_subgraph(A_hat, max_nodes=max_nodes)
    A_sub = A_hat[idx][:, idx]
    X_sub = X[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx_full["X_proj"][idx]}
    Z_sub = reconverge(model, Z_full[idx].clone(), ctx_sub)
    return X_sub, A_sub, Z_sub, ctx_sub, ctx_full, Z_full, idx


def full_graph_ctx_Z(model, X, A_hat):
    """For full-graph analysis: returns (ctx, Z_star)."""
    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)
    return ctx, Z_star
