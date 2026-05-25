"""Scope extension: AEGIS-style analysis on explicit GNNs.

Shows that the S_c framework applies beyond IGNN by computing the
"unrolled sensitivity" dZ_K/dA for K-layer explicit GNNs via finite
differences. Vulnerability rankings and attack directions transfer.

Models tested: IGNN, GCN-2, GCN-4, GIN-2, GAT-2, GraphSAGE-2, APPNP
All on Cora, 10 seeds.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_explicit_gnn_extension
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Explicit GNN architectures
# ---------------------------------------------------------------

class ExplicitGCN(nn.Module):
    """Standard K-layer GCN with ReLU and dropout."""

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
    """Graph Isomorphism Network (Xu et al. 2019), adapted for normalized A_hat.

    Since A_hat = D^{-1/2}(A+I)D^{-1/2} already includes self-loops,
    the aggregation is: MLP((1+eps) * A_hat @ Z) which scales the
    normalized neighborhood sum (including self).
    """

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
            # A_hat already includes self-loops; eps scales the combined signal
            agg = (1 + self.eps_vals[k]) * (A_hat @ Z)
            Z = mlp(agg)
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitGAT(nn.Module):
    """Edge-weighted GAT with multi-head attention. Attention scores are
    modulated by A_hat values so that edge weights participate continuously
    in the computation, enabling finite-difference sensitivity analysis."""

    def __init__(self, n_features, hidden, n_classes, n_layers=2,
                 n_heads=8, dropout=0.6, attn_dropout=0.6):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        head_dim = hidden  # each head outputs `hidden` dims
        self.projs = nn.ModuleList()
        self.attn_src = nn.ParameterList()
        self.attn_dst = nn.ParameterList()
        in_dim = n_features
        for layer_i in range(n_layers):
            self.projs.append(nn.Linear(in_dim, n_heads * head_dim, bias=False))
            self.attn_src.append(nn.Parameter(torch.randn(n_heads, head_dim, 1) * 0.01))
            self.attn_dst.append(nn.Parameter(torch.randn(n_heads, head_dim, 1) * 0.01))
            in_dim = n_heads * head_dim
        # hidden state is n_heads * head_dim from last layer
        self.head = nn.Linear(n_heads * head_dim, n_classes)

    def forward_hidden(self, X, A_hat):
        N = X.shape[0]
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for k, proj in enumerate(self.projs):
            Z_proj = proj(Z)  # (N, n_heads * head_dim)
            head_dim = Z_proj.shape[1] // self.n_heads
            Z_heads = Z_proj.view(N, self.n_heads, head_dim)  # (N, H, D)

            # Compute attention per head: (H, N, 1)
            e_src = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_src[k]).squeeze(-1)  # (N, H)
            e_dst = torch.einsum('nhd,hdk->nhk', Z_heads, self.attn_dst[k]).squeeze(-1)  # (N, H)

            # attn_logits: (H, N, N)
            attn_logits = e_src.T.unsqueeze(2) + e_dst.T.unsqueeze(1)  # (H, N, N)

            mask = (A_hat.abs() < 1e-10).unsqueeze(0)  # (1, N, N)
            attn_logits = attn_logits.masked_fill(mask, -1e9)
            attn_weights = F_func.softmax(attn_logits, dim=2)  # (H, N, N)
            attn_weights = F_func.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            attn_weights = attn_weights * A_hat.unsqueeze(0)  # edge-weight modulation

            # Aggregate: (H, N, N) @ (N, H, D) -> need per-head
            # Z_heads: (N, H, D) -> (H, N, D)
            Z_h = Z_heads.permute(1, 0, 2)  # (H, N, D)
            out = torch.bmm(attn_weights, Z_h)  # (H, N, D)
            # Concatenate heads: (N, H*D)
            Z = out.permute(1, 0, 2).contiguous().view(N, -1)
            Z = F_func.elu(Z)
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitGraphSAGE(nn.Module):
    """GraphSAGE with mean aggregation. Concatenates self + neighbor mean."""

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
    """APPNP: MLP feature transform then K steps of personalized PageRank propagation."""

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
# Analysis functions
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


def brute_force_ranking(model, X_sub, A_sub, edge_list):
    with torch.no_grad():
        Z_base = model.forward_hidden(X_sub, A_sub)
    shifts = []
    with torch.no_grad():
        for i, j in edge_list:
            A_pert = A_sub.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            Z_pert = model.forward_hidden(X_sub, A_pert)
            shifts.append(float((Z_pert - Z_base).norm()))
    return shifts


# Model-specific hyperparameters from literature
_HP = {
    "GCN-2":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "GCN-4":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "GIN-2":  {"lr": 0.005, "wd": 5e-4, "epochs": 400},
    "GAT-2":  {"lr": 0.005, "wd": 5e-4, "epochs": 400},
    "SAGE-2": {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "APPNP":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
}


def run_single(model_name, model, X, A_hat, y, train_mask, val_mask, seed, device,
               test_mask=None):
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
        # Check validation every epoch for better model selection
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

    # Compute full test-set accuracy (standard metric)
    test_acc = None
    if test_mask is not None:
        with torch.no_grad():
            logits_test, _ = model(X, A_hat)
            test_acc = float((logits_test.argmax(1)[test_mask] == y[test_mask]).float().mean())

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    X_sub = X[idx]

    S, Z_base = compute_explicit_sensitivity(model, X_sub, A_sub)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
    sigma1 = float(sigma_c[0])

    eps = 0.01
    predicted = eps * sigma1
    dA = torch.zeros_like(A_sub)
    weights = eps * Vh_c[0]
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    with torch.no_grad():
        Z_pert = model.forward_hidden(X_sub, A_sub + dA).reshape(-1)
    actual = float((Z_pert - Z_base).norm())
    tightness = actual / predicted if predicted > 1e-12 else float("nan")

    with torch.no_grad():
        rand_weights = torch.randn(len(edge_list), device=A_sub.device)
        rand_weights *= eps / rand_weights.norm()
        rand_dA = torch.zeros_like(A_sub)
        for k, (i, j) in enumerate(edge_list):
            rand_dA[i, j] = float(rand_weights[k])
            rand_dA[j, i] = float(rand_weights[k])
        Z_rand = model.forward_hidden(X_sub, A_sub + rand_dA).reshape(-1)
    rand_shift = float((Z_rand - Z_base).norm())
    atk_adv = actual / rand_shift if rand_shift > 1e-12 else float("nan")

    aegis_vuln = [(i, j, float(S_c[:, k].norm())) for k, (i, j) in enumerate(edge_list)]
    aegis_vuln.sort(key=lambda x: x[2], reverse=True)

    bf_shifts = brute_force_ranking(model, X_sub, A_sub, edge_list)
    bf_ranking = sorted(
        [(edge_list[k][0], edge_list[k][1], bf_shifts[k]) for k in range(len(edge_list))],
        key=lambda x: x[2], reverse=True,
    )
    bf_dict = {(min(i, j), max(i, j)): rank for rank, (i, j, _) in enumerate(bf_ranking)}
    common = []
    for rank, (i, j, _) in enumerate(aegis_vuln):
        key = (min(i, j), max(i, j))
        if key in bf_dict:
            common.append((rank, bf_dict[key]))
    tau = kendalltau(*zip(*common))[0] if len(common) >= 3 else None

    with torch.no_grad():
        logits_sub, _ = model(X_sub, A_sub)
    y_sub = y[idx]
    acc = float((logits_sub.argmax(1) == y_sub).float().mean())

    return {"tightness": tightness, "atk_adv": atk_adv, "tau": tau,
            "acc": acc, "test_acc": test_acc}


def run_ignn_single(data, seed, device):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float((logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    from iem.adversarial import _compute_structural_jacobian, structural_sensitivity_matrix
    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
    sigma1 = float(sigma_c[0])
    eps = 0.01
    predicted = eps * sigma1
    dA = torch.zeros_like(A_sub)
    weights = eps * Vh_c[0]
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_pert)
            if (Z_new - Z).norm() < 1e-8:
                break
            Z = Z_new
    actual = float((Z_new - Z_sub).norm())
    tightness = actual / predicted if predicted > 1e-12 else float("nan")

    with torch.no_grad():
        rand_dA = torch.randn_like(A_sub)
        rand_dA = (rand_dA + rand_dA.T) / 2
        rand_dA *= eps / rand_dA.norm()
        ctx_rand = {**ctx_sub, "A_hat": A_sub + rand_dA}
        Z = Z_sub.clone()
        for _ in range(100):
            Z_new = model.operator(Z, ctx_rand)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    rand_shift = float((Z_new - Z_sub).norm())
    atk_adv = actual / rand_shift if rand_shift > 1e-12 else float("nan")

    aegis_vuln = [(i, j, float(S_c[:, k].norm())) for k, (i, j) in enumerate(edge_list)]
    aegis_vuln.sort(key=lambda x: x[2], reverse=True)
    bf_shifts = []
    with torch.no_grad():
        for i, j in edge_list:
            A_p = A_sub.clone(); A_p[i,j] = 0.0; A_p[j,i] = 0.0
            ctx_bf = {**ctx_sub, "A_hat": A_p}
            Z = Z_sub.clone()
            for _ in range(50):
                Z = model.operator(Z, ctx_bf)
            bf_shifts.append(float((Z - Z_sub).norm()))
    bf_ranking = sorted([(edge_list[k][0], edge_list[k][1], bf_shifts[k]) for k in range(len(edge_list))], key=lambda x: x[2], reverse=True)
    bf_dict = {(min(i,j), max(i,j)): rank for rank, (i,j,_) in enumerate(bf_ranking)}
    common = []
    for rank, (i,j,_) in enumerate(aegis_vuln):
        key = (min(i,j), max(i,j))
        if key in bf_dict:
            common.append((rank, bf_dict[key]))
    tau = kendalltau(*zip(*common))[0] if len(common) >= 3 else None

    y_sub = y[idx]
    logits_sub = model.head(Z_sub)
    acc = float((logits_sub.argmax(1) == y_sub).float().mean())

    return {"tightness": tightness, "atk_adv": atk_adv, "tau": tau, "acc": acc}


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None and not np.isnan(v)]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    data = _load_cora(Path("datasets/cora"))
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    nf, nc = data["n_features"], data["n_classes"]

    models_to_test = [
        ("IGNN",       None),
        ("GCN-2",      lambda: ExplicitGCN(nf, 64, nc, n_layers=2, dropout=0.5).to(device)),
        ("GCN-4",      lambda: ExplicitGCN(nf, 64, nc, n_layers=4, dropout=0.5).to(device)),
        ("GIN-2",      lambda: ExplicitGIN(nf, 128, nc, n_layers=2, dropout=0.5).to(device)),
        ("GAT-2",      lambda: ExplicitGAT(nf, 8, nc, n_layers=2, n_heads=8,
                                            dropout=0.6, attn_dropout=0.6).to(device)),
        ("SAGE-2",     lambda: ExplicitGraphSAGE(nf, 64, nc, n_layers=2, dropout=0.5).to(device)),
        ("APPNP",      lambda: ExplicitAPPNP(nf, 64, nc, n_prop=10, alpha=0.1, dropout=0.5).to(device)),
    ]

    all_results = {name: [] for name, _ in models_to_test}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for model_name, model_fn in models_to_test:
            if model_name == "IGNN":
                r = run_ignn_single(data, seed, device)
            else:
                model = model_fn()
                r = run_single(model_name, model, X, A_hat, y,
                              data["train_mask"], data["val_mask"], seed, device,
                              test_mask=data["test_mask"])
            if r:
                all_results[model_name].append(r)
                tau_s = f"{r['tau']:+.3f}" if r['tau'] is not None else "N/A"
                ta = r.get('test_acc')
                ta_s = f"{ta:.3f}" if ta is not None else "N/A"
                print(f"  {model_name:<8}: tight={r['tightness']:.3f} adv={r['atk_adv']:.1f}x tau={tau_s} acc={r['acc']:.3f} test_acc={ta_s}", flush=True)
            else:
                print(f"  {model_name:<8}: SKIP", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 100)
    print("SCOPE EXTENSION: AEGIS on Multiple GNN Architectures (Cora, 10 seeds)")
    print("=" * 100)
    print(f"{'Model':<10} {'Tightness':>14} {'AtkAdv':>12} {'Kendall τ':>14} {'SubgAcc':>12} {'TestAcc':>14}")
    print("-" * 110)
    for model_name, _ in models_to_test:
        rs = all_results[model_name]
        test_accs = [r.get('test_acc') for r in rs if r.get('test_acc') is not None]
        print(f"{model_name:<10} "
              f"{agg([r['tightness'] for r in rs]):>14} "
              f"{agg([r['atk_adv'] for r in rs]):>12} "
              f"{agg([r['tau'] for r in rs]):>14} "
              f"{agg([r['acc'] for r in rs]):>12} "
              f"{agg(test_accs):>14}")

    results_path = Path("docs/exp_explicit_gnn_extension_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Scope Extension: AEGIS on Multiple GNN Architectures (Cora, 10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Model | Tightness | AtkAdv | Kendall τ | SubgAcc | TestAcc |\n")
        f.write("|---|---|---|---|---|---|\n")
        for model_name, _ in models_to_test:
            rs = all_results[model_name]
            test_accs = [r.get('test_acc') for r in rs if r.get('test_acc') is not None]
            f.write(f"| {model_name} "
                    f"| {agg([r['tightness'] for r in rs])} "
                    f"| {agg([r['atk_adv'] for r in rs])} "
                    f"| {agg([r['tau'] for r in rs])} "
                    f"| {agg([r['acc'] for r in rs])} "
                    f"| {agg(test_accs)} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
