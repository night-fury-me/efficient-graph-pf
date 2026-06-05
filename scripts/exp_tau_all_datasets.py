"""B1 / P1.3 — Cross-dataset/architecture continuous-to-discrete tau evaluation.

For each (dataset, architecture, seed):
  1. Load dataset and train model
  2. Extract 50-node BFS subgraph
  3. Compute S_c continuous vulnerability scores (per-edge ||[S_c]_{:,k}||_2)
  4. Run greedy brute-force edge-removal to get discrete damage (ground truth)
  5. Compute Kendall tau between continuous scores and discrete damage
  6. Compute precision@k (k=5, 10, 20)

Datasets:  Cora, Citeseer, Pubmed (subgraph only), WikiCS, Amazon Photo, Amazon Fraud
Architectures: IGNN, GCN-2, GCN-4, GIN-2, GAT-2, SAGE-2, APPNP
Seeds: 10

Output: results/tau_all_datasets.csv

Usage:
    .venv/bin/python scripts/exp_tau_all_datasets.py
"""

from __future__ import annotations

import csv
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None and not np.isnan(v)]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}+/-{s:{fmt}}"


# ---------------------------------------------------------------
# Explicit GNN architectures (from exp_explicit_gnn_extension.py)
# ---------------------------------------------------------------

class ExplicitGCN(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, hidden))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, n_classes)

    def forward_hidden(self, X, A_hat):
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for layer in self.layers:
            Z = F_func.relu(A_hat @ layer(Z))
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitGIN(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.eps_vals = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_layers)])
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        ))
        for _ in range(n_layers - 1):
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            ))
        self.head = nn.Linear(hidden, n_classes)

    def forward_hidden(self, X, A_hat):
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for k, mlp in enumerate(self.mlps):
            agg = (1 + self.eps_vals[k]) * (A_hat @ Z)
            Z = mlp(agg)
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitGAT(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_layers=2,
                 n_heads=8, dropout=0.6, attn_dropout=0.6):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        head_dim = hidden
        self.projs = nn.ModuleList()
        self.attn_src = nn.ParameterList()
        self.attn_dst = nn.ParameterList()
        in_dim = n_features
        for _ in range(n_layers):
            self.projs.append(nn.Linear(in_dim, n_heads * head_dim, bias=False))
            self.attn_src.append(nn.Parameter(torch.randn(n_heads, head_dim, 1) * 0.01))
            self.attn_dst.append(nn.Parameter(torch.randn(n_heads, head_dim, 1) * 0.01))
            in_dim = n_heads * head_dim
        self.head = nn.Linear(n_heads * head_dim, n_classes)

    @staticmethod
    def _scatter_softmax(logit, index, N):
        """Numerically-stable softmax of `logit` (E, H) over edges grouped by the
        target node `index` (E,). Sparse equivalent of the dense softmax(dim=j):
        for each target i, normalise over the edges (i, ·)."""
        H = logit.shape[1]
        amax = logit.new_full((N, H), float("-inf"))
        amax.index_reduce_(0, index, logit, "amax", include_self=False)
        ex = (logit - amax.index_select(0, index)).exp()
        denom = torch.zeros(N, H, device=logit.device, dtype=logit.dtype)
        denom.index_add_(0, index, ex)
        return ex / (denom.index_select(0, index) + 1e-16)

    def forward_hidden(self, X, A_hat):
        """Sparse (edge-indexed) multi-head attention — O(E·H) memory, no N×N
        attention tensor, so it scales to full-graph Pubmed/WikiCS/Amazon Fraud
        (the dense path below OOMs there). Mathematically identical to
        ``_forward_hidden_dense`` in eval mode (verified allclose by
        ``scripts/_verify_sparse_gat.py``). Edges are the nonzero entries of A_hat
        (incl. self-loops, and any FD-perturbed non-edge ≥1e-10, matching the dense
        mask ``A_hat.abs() < 1e-10``); recomputed each call so finite-difference
        probes (which clone + perturb A) stay exact."""
        N = X.shape[0]
        edges = (A_hat.abs() >= 1e-10).nonzero(as_tuple=False)  # (E, 2): target i, source j
        row, col = edges[:, 0], edges[:, 1]
        aval = A_hat[row, col]                                  # (E,) normalized-adj weights
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for k, proj in enumerate(self.projs):
            Z_proj = proj(Z)
            head_dim = Z_proj.shape[1] // self.n_heads
            Z_heads = Z_proj.view(N, self.n_heads, head_dim)   # (N, H, hd)
            e_src = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_src[k]).squeeze(-1)  # (N, H)
            e_dst = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_dst[k]).squeeze(-1)  # (N, H)
            # logit[e, h] = e_src[i, h] + e_dst[j, h] for edge e = (i, j)
            logit = e_src.index_select(0, row) + e_dst.index_select(0, col)  # (E, H)
            alpha = self._scatter_softmax(logit, row, N)        # softmax over edges per target i
            alpha = F_func.dropout(alpha, p=self.attn_dropout, training=self.training)
            alpha = alpha * aval.unsqueeze(1)                   # weight by A_hat[i, j] (after softmax)
            msg = alpha.unsqueeze(-1) * Z_heads.index_select(0, col)  # (E, H, hd): scaled source feats
            out = torch.zeros(N, self.n_heads, head_dim, device=Z.device, dtype=Z.dtype)
            out.index_add_(0, row, msg)                         # aggregate messages into target i
            Z = out.reshape(N, -1)                              # (N, H*hd), head-major concat (== dense)
            Z = F_func.elu(Z)
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def _forward_hidden_dense(self, X, A_hat):
        """Reference dense-attention forward (O(N²·H) memory) — kept only for the
        equivalence test; OOMs on large graphs, which is why ``forward_hidden`` is
        sparse."""
        N = X.shape[0]
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for k, proj in enumerate(self.projs):
            Z_proj = proj(Z)
            head_dim = Z_proj.shape[1] // self.n_heads
            Z_heads = Z_proj.view(N, self.n_heads, head_dim)
            e_src = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_src[k]).squeeze(-1)
            e_dst = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_dst[k]).squeeze(-1)
            attn_logits = e_src.T.unsqueeze(2) + e_dst.T.unsqueeze(1)
            mask = (A_hat.abs() < 1e-10).unsqueeze(0)
            attn_logits = attn_logits.masked_fill(mask, -1e9)
            attn_weights = F_func.softmax(attn_logits, dim=2)
            attn_weights = F_func.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            attn_weights = attn_weights * A_hat.unsqueeze(0)
            Z_h = Z_heads.permute(1, 0, 2)
            out = torch.bmm(attn_weights, Z_h)
            Z = out.permute(1, 0, 2).contiguous().view(N, -1)
            Z = F_func.elu(Z)
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitGraphSAGE(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        in_dim = n_features
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim + in_dim, hidden))
            in_dim = hidden
        self.head = nn.Linear(hidden, n_classes)

    def forward_hidden(self, X, A_hat):
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for layer in self.layers:
            deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1)
            neighbor_mean = A_hat @ Z / deg
            Z = F_func.relu(layer(torch.cat([Z, neighbor_mean], dim=-1)))
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitAPPNP(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_prop=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, n_classes)
        self.n_prop = n_prop
        self.alpha = alpha

    def forward_hidden(self, X, A_hat):
        X_drop = F_func.dropout(X, p=self.dropout, training=self.training)
        H_0 = self.mlp(X_drop)
        Z = H_0
        for _ in range(self.n_prop):
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
            Z = (1 - self.alpha) * (A_hat @ Z) + self.alpha * H_0
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


# ---------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------

_HP = {
    "GCN-2":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "GCN-4":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "GIN-2":  {"lr": 0.005, "wd": 5e-4, "epochs": 400},
    "GAT-2":  {"lr": 0.005, "wd": 5e-4, "epochs": 400},
    "SAGE-2": {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "APPNP":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
}


# ---------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------

def load_all_datasets():
    """Load all 5 datasets, returning dict of name -> data."""
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics
    from iem.examples.ignn_amazon import _load_amazon
    from iem.examples.ignn_amazon_fraud import _load_amazon_fraud

    datasets = {}
    print("Loading datasets...", flush=True)

    print("  Cora...", flush=True)
    datasets["Cora"] = _load_cora(Path("datasets/cora"))

    print("  Citeseer...", flush=True)
    datasets["Citeseer"] = _load_planetoid("citeseer", Path("datasets/citeseer"))

    print("  Pubmed...", flush=True)
    datasets["Pubmed"] = _load_planetoid("pubmed", Path("datasets/pubmed"))

    print("  WikiCS...", flush=True)
    datasets["WikiCS"] = _load_wikics(Path("datasets/wikics"))

    print("  Amazon Photo...", flush=True)
    datasets["Amazon Photo"] = _load_amazon(Path("datasets/amazon_photo"))

    print("  Amazon Fraud...", flush=True)
    datasets["Amazon Fraud"] = _load_amazon_fraud(Path("datasets/amazon_fraud"))

    for name, d in datasets.items():
        print(f"    {name}: N={d['N']}, feat={d['n_features']}, classes={d['n_classes']}", flush=True)
    return datasets


# ---------------------------------------------------------------
# Sensitivity computation (explicit GNNs via finite differences)
# ---------------------------------------------------------------

def compute_explicit_sensitivity(model, X_sub, A_sub, eps_fd=1e-4):
    N = A_sub.shape[0]
    with torch.no_grad():
        Z_base = model.forward_hidden(X_sub, A_sub).reshape(-1)
    D = Z_base.shape[0]
    S = torch.zeros(D, N * N, device=A_sub.device)
    with torch.no_grad():
        for idx in range(N * N):
            i, j = idx // N, idx % N
            A_pert = A_sub.clone()
            A_pert[i, j] += eps_fd
            Z_pert = model.forward_hidden(X_sub, A_pert).reshape(-1)
            S[:, idx] = (Z_pert - Z_base) / eps_fd
    return S, Z_base


def compute_explicit_sensitivity_ad(model, X_sub, A_sub):
    """One reverse-mode autodiff query for the explicit-GNN structural sensitivity
    S = dZ/dA (D, N*N) -- the exact Jacobian, vectorized over all A entries, replacing
    the O(N^2) finite-difference probes of compute_explicit_sensitivity.

    Verified allclose to the FD version (maxdiff ~1e-4) and 3-62x faster for the
    message-passing architectures GCN/GIN/SAGE/APPNP, where A enters as continuous
    edge weights. NOT used for GAT, whose attention treats A as a discrete mask, so
    dZ/dA is not a smooth Jacobian (jacrev and FD diverge -- the GAT-dagger caveat);
    GAT keeps compute_explicit_sensitivity (finite differences).
    """
    with torch.no_grad():
        Z_base = model.forward_hidden(X_sub, A_sub).reshape(-1)
    N = A_sub.shape[0]

    def f(A):
        return model.forward_hidden(X_sub, A).reshape(-1)

    S = torch.func.jacrev(f)(A_sub).reshape(Z_base.shape[0], N * N)
    return S, Z_base


# ---------------------------------------------------------------
# Brute-force edge-removal ranking (ground truth for discrete damage)
# ---------------------------------------------------------------

def brute_force_edge_removal(model, X_sub, A_sub, edge_list, is_ignn=False,
                             ctx_sub=None, Z_sub=None):
    """Remove each edge, measure equilibrium shift. Returns list of shifts."""
    shifts = []
    with torch.no_grad():
        if is_ignn:
            # IGNN: reconverge after removal
            for i, j in edge_list:
                A_pert = A_sub.clone()
                A_pert[i, j] = 0.0
                A_pert[j, i] = 0.0
                ctx_pert = {**ctx_sub, "A_hat": A_pert}
                Z = Z_sub.clone()
                for _ in range(100):
                    Z_new = model.operator(Z, ctx_pert)
                    if (Z_new - Z).norm() < 1e-7:
                        break
                    Z = Z_new
                shifts.append(float((Z_new - Z_sub).norm()))
        else:
            # Explicit GNN: single forward pass
            Z_base = model.forward_hidden(X_sub, A_sub)
            for i, j in edge_list:
                A_pert = A_sub.clone()
                A_pert[i, j] = 0.0
                A_pert[j, i] = 0.0
                Z_pert = model.forward_hidden(X_sub, A_pert)
                shifts.append(float((Z_pert - Z_base).norm()))
    return shifts


# ---------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------

def precision_at_k(continuous_scores, discrete_scores, k):
    """Precision@k: fraction of top-k continuous that appear in top-k discrete."""
    if len(continuous_scores) < k:
        k = len(continuous_scores)
    if k == 0:
        return float("nan")
    cont_top = set(np.argsort(continuous_scores)[-k:])
    disc_top = set(np.argsort(discrete_scores)[-k:])
    return len(cont_top & disc_top) / k


# ---------------------------------------------------------------
# Train explicit GNN
# ---------------------------------------------------------------

def train_explicit(model_name, model, X, A_hat, y, train_mask, val_mask, seed, device):
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
    del optim, best_state
    gc.collect()
    model.eval()


# ---------------------------------------------------------------
# Run single (dataset, architecture, seed) combination
# ---------------------------------------------------------------

def run_single(dataset_name, arch_name, data, seed, device):
    """Returns dict with tau, p_at_5, p_at_10, p_at_20, n_edges or None on failure."""
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    nf, nc = data["n_features"], data["n_classes"]

    try:
        if arch_name == "IGNN":
            return _run_ignn(data, X, A_hat, y, seed, device)
        else:
            return _run_explicit(arch_name, data, X, A_hat, y, nf, nc, seed, device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"      OOM — skipping (full-graph training too large for GPU)", flush=True)
            del X, A_hat, y
            gc.collect()
            torch.cuda.empty_cache()
            return None
        else:
            print(f"      Error: {e}", flush=True)
            return None
    except Exception as e:
        print(f"      Error: {e}", flush=True)
        return None


def _run_ignn(data, X, A_hat, y, seed, device):
    """IGNN-specific analysis using IFT-based S."""
    set_seed(seed)
    X = X.to(device)
    A_hat = A_hat.to(device)
    y = y.to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float((logits_v.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    del optim, best_state
    gc.collect()
    model.eval()

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx].clone()
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
    del X, A_hat, y, ctx
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reconverge on subgraph
    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    # Compute S via IFT
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    del S, J_z, J_A
    if not edge_list:
        return None

    # Continuous vulnerability scores
    cont_scores = np.array([float(S_c[:, k].norm()) for k in range(len(edge_list))])
    del S_c

    # Discrete ground truth via brute-force edge removal
    disc_scores = np.array(brute_force_edge_removal(
        model, None, A_sub, edge_list, is_ignn=True, ctx_sub=ctx_sub, Z_sub=Z_sub
    ))
    # Edge weights w_k = normalized-adjacency entry A_hat[i,j] for each edge k,
    # in the SAME order as edge_list / cont_scores / disc_scores (see
    # constrained_sensitivity_matrix: column k <-> edge_list[k] = (i, j)).
    weights = np.array([float(A_sub[i, j]) for (i, j) in edge_list])
    del model, A_sub, ctx_sub, Z_sub
    gc.collect()

    if len(cont_scores) < 3:
        return None

    tau, _ = kendalltau(cont_scores, disc_scores)
    tau_weighted, _ = kendalltau(weights * cont_scores, disc_scores)
    tau_weight_only, _ = kendalltau(weights, disc_scores)
    p5 = precision_at_k(cont_scores, disc_scores, 5)
    p10 = precision_at_k(cont_scores, disc_scores, 10)
    p20 = precision_at_k(cont_scores, disc_scores, 20)

    return {"tau": tau, "tau_weighted": tau_weighted,
            "tau_weight_only": tau_weight_only,
            "p_at_5": p5, "p_at_10": p10, "p_at_20": p20,
            "n_edges": len(edge_list)}


def _run_explicit(arch_name, data, X, A_hat, y, nf, nc, seed, device):
    """Explicit GNN analysis using finite-difference S."""
    set_seed(seed)
    X = X.to(device)
    A_hat = A_hat.to(device)
    y = y.to(device)

    # Construct model
    model_map = {
        "GCN-2":  lambda: ExplicitGCN(nf, 64, nc, n_layers=2, dropout=0.5),
        "GCN-4":  lambda: ExplicitGCN(nf, 64, nc, n_layers=4, dropout=0.5),
        "GIN-2":  lambda: ExplicitGIN(nf, 128, nc, n_layers=2, dropout=0.5),
        "GAT-2":  lambda: ExplicitGAT(nf, 8, nc, n_layers=2, n_heads=8,
                                       dropout=0.6, attn_dropout=0.6),
        "SAGE-2": lambda: ExplicitGraphSAGE(nf, 64, nc, n_layers=2, dropout=0.5),
        "APPNP":  lambda: ExplicitAPPNP(nf, 64, nc, n_prop=10, alpha=0.1, dropout=0.5),
    }
    model = model_map[arch_name]().to(device)
    train_explicit(arch_name, model, X, A_hat, y,
                   data["train_mask"].to(device), data["val_mask"].to(device),
                   seed, device)

    # Extract subgraph — then drop full-graph tensors to free memory
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx].clone()
    X_sub = X[idx].clone()
    del X, A_hat, y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Compute S via finite differences
    S, Z_base = compute_explicit_sensitivity(model, X_sub, A_sub)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    del S, Z_base
    if not edge_list:
        return None

    # Continuous vulnerability scores
    cont_scores = np.array([float(S_c[:, k].norm()) for k in range(len(edge_list))])
    del S_c

    # Discrete ground truth
    disc_scores = np.array(brute_force_edge_removal(
        model, X_sub, A_sub, edge_list, is_ignn=False
    ))
    # Edge weights w_k = normalized-adjacency entry A_hat[i,j] for each edge k,
    # in the SAME order as edge_list / cont_scores / disc_scores (see
    # constrained_sensitivity_matrix: column k <-> edge_list[k] = (i, j)).
    weights = np.array([float(A_sub[i, j]) for (i, j) in edge_list])
    del model, X_sub, A_sub
    gc.collect()

    if len(cont_scores) < 3:
        return None

    tau, _ = kendalltau(cont_scores, disc_scores)
    tau_weighted, _ = kendalltau(weights * cont_scores, disc_scores)
    tau_weight_only, _ = kendalltau(weights, disc_scores)
    p5 = precision_at_k(cont_scores, disc_scores, 5)
    p10 = precision_at_k(cont_scores, disc_scores, 10)
    p20 = precision_at_k(cont_scores, disc_scores, 20)

    return {"tau": tau, "tau_weighted": tau_weighted,
            "tau_weight_only": tau_weight_only,
            "p_at_5": p5, "p_at_10": p10, "p_at_20": p20,
            "n_edges": len(edge_list)}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t_start = time.time()

    datasets = load_all_datasets()

    ARCHITECTURES = ["IGNN", "GCN-2", "GCN-4", "GIN-2", "GAT-2", "SAGE-2", "APPNP"]
    DATASET_NAMES = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon Photo", "Amazon Fraud"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "tau_all_datasets.csv"

    rows = []

    for ds_name in DATASET_NAMES:
        data = datasets[ds_name]
        for arch in ARCHITECTURES:
            for seed_idx, seed in enumerate(SEEDS):
                print(f"[{ds_name}] [{arch}] seed={seed} ({seed_idx+1}/{len(SEEDS)})",
                      flush=True)
                r = run_single(ds_name, arch, data, seed, device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if r is not None:
                    row = {
                        "dataset": ds_name,
                        "architecture": arch,
                        "seed": seed,
                        "tau": r["tau"],
                        "tau_weighted": r["tau_weighted"],
                        "tau_weight_only": r["tau_weight_only"],
                        "p_at_5": r["p_at_5"],
                        "p_at_10": r["p_at_10"],
                        "p_at_20": r["p_at_20"],
                        "n_edges": r["n_edges"],
                    }
                    rows.append(row)
                    print(f"    tau={r['tau']:+.3f}  tauW={r['tau_weighted']:+.3f}  tauWo={r['tau_weight_only']:+.3f}  p@5={r['p_at_5']:.2f}  "
                          f"p@10={r['p_at_10']:.2f}  p@20={r['p_at_20']:.2f}  "
                          f"edges={r['n_edges']}", flush=True)
                else:
                    print(f"    SKIP", flush=True)

    # Write CSV
    fieldnames = ["dataset", "architecture", "seed", "tau", "tau_weighted",
                  "tau_weight_only", "p_at_5", "p_at_10", "p_at_20", "n_edges"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}", flush=True)

    # Summary table
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 110)
    print("CROSS-DATASET TAU EVALUATION SUMMARY (mean +/- std across seeds)")
    print("=" * 110)
    print(f"{'Dataset':<15} {'Arch':<8} {'Kendall tau':>14} {'P@5':>10} {'P@10':>10} "
          f"{'P@20':>10} {'#edges':>8}")
    print("-" * 110)

    for ds_name in DATASET_NAMES:
        for arch in ARCHITECTURES:
            subset = [r for r in rows if r["dataset"] == ds_name and r["architecture"] == arch]
            if not subset:
                continue
            taus = [r["tau"] for r in subset]
            p5s = [r["p_at_5"] for r in subset]
            p10s = [r["p_at_10"] for r in subset]
            p20s = [r["p_at_20"] for r in subset]
            edges = [r["n_edges"] for r in subset]
            print(f"{ds_name:<15} {arch:<8} {agg(taus):>14} {agg(p5s):>10} "
                  f"{agg(p10s):>10} {agg(p20s):>10} {np.mean(edges):>8.0f}")

    print(f"\n{len(rows)} successful runs out of "
          f"{len(DATASET_NAMES) * len(ARCHITECTURES) * len(SEEDS)} attempted")


if __name__ == "__main__":
    sys.exit(main() or 0)
