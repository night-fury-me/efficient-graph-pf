"""
Round 3 remaining experiments:
1. GNNExplainer-style edge attribution comparison (P1.3)
2. Degree-weighted S_c ablation (R2-P2-10)
3. Scalability beyond N=200 (R3-P2-8)
4. Full-graph task accuracy for all models (added to explicit table)

All experiments: 10 seeds on Cora.
Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
"""

from __future__ import annotations
import sys
import random
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
from iem.examples.exp_explicit_gnn_extension import (
    ExplicitGCN, ExplicitGIN, ExplicitGAT, ExplicitGraphSAGE, ExplicitAPPNP,
    compute_explicit_sensitivity, set_seed, brute_force_ranking,
)

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def load_cora():
    data = _load_cora(Path("datasets/cora"))
    return data


def train_model(model_cls, data, seed, device, is_ignn=False, **kwargs):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = model_cls(data["n_features"], 64, data["n_classes"], **kwargs).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        if is_ignn:
            logits, _, _ = model(X, A_hat)
        else:
            logits, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if is_ignn:
                    logits_v, _, _ = model(X, A_hat)
                else:
                    logits_v, _ = model(X, A_hat)
                val_acc = float((logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if is_ignn:
            logits_test, _, _ = model(X, A_hat)
        else:
            logits_test, _ = model(X, A_hat)
    test_acc = float((logits_test.argmax(1)[data["test_mask"]] == y[data["test_mask"]]).float().mean())

    return model, test_acc


# ---------------------------------------------------------------
# Experiment 1: GNNExplainer-style edge attribution comparison
# ---------------------------------------------------------------

def gradient_edge_attribution(model, X_sub, A_sub, is_ignn=False):
    """
    Compute gradient-based edge attribution (similar to GNNExplainer's
    gradient signal). For each edge (i,j), compute |dL/dA_ij| where L
    is the prediction entropy on the subgraph.
    """
    A_sub_param = A_sub.clone().detach().requires_grad_(True)
    if is_ignn:
        logits, _, _ = model(X_sub, A_sub_param)
    else:
        logits, _ = model(X_sub, A_sub_param)
    probs = F_func.softmax(logits, dim=1)
    entropy = -(probs * (probs + 1e-10).log()).sum()
    entropy.backward()
    grad = A_sub_param.grad.abs()
    edge_grad = (grad + grad.T) / 2
    return edge_grad


def brute_force_ranking_generic(model, X_sub, A_sub, edge_list, is_ignn=False, ctx_sub=None, Z_sub=None):
    """Brute-force ranking that works for both explicit and IGNN models."""
    if not is_ignn:
        return brute_force_ranking(model, X_sub, A_sub, edge_list)

    shifts = []
    with torch.no_grad():
        for i, j in edge_list:
            A_p = A_sub.clone()
            A_p[i, j] = 0.0
            A_p[j, i] = 0.0
            ctx_bf = {**ctx_sub, "A_hat": A_p}
            Z = Z_sub.clone()
            for _ in range(50):
                Z_new = model.operator(Z, ctx_bf)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
            shifts.append(float((Z - Z_sub).norm()))
    return shifts


def run_explainer_comparison(data, device):
    """Compare AEGIS vs gradient-based edge attribution for vulnerability ranking."""
    print("\n" + "=" * 60)
    print("Experiment 1: AEGIS vs Gradient Edge Attribution (10 seeds)")
    print("=" * 60)

    models_cfg = {
        'GCN-2': (ExplicitGCN, {'n_layers': 2}, False),
        'GIN-2': (ExplicitGIN, {'n_layers': 2}, False),
        'IGNN':  (IGNN, {}, True),
    }

    for model_name, (cls, kwargs, is_ignn) in models_cfg.items():
        tau_aegis_list = []
        tau_grad_list = []
        overlap_at5 = []

        for seed in SEEDS:
            model, _ = train_model(cls, data, seed, device, is_ignn=is_ignn, **kwargs)
            X = data["X"].to(device)
            A_hat = data["A_hat"].to(device)
            y = data["y"].to(device)

            idx = extract_ego_subgraph(A_hat, max_nodes=50)
            A_sub = A_hat[idx][:, idx]
            X_sub = X[idx]

            ctx_sub = None
            Z_sub = None
            if is_ignn:
                from iem.adversarial import _compute_structural_jacobian, structural_sensitivity_matrix
                with torch.no_grad():
                    _, Z_star, ctx = model(X, A_hat)
                Z_sub = Z_star[idx].clone()
                ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
                with torch.no_grad():
                    for _ in range(200):
                        Z_new = model.operator(Z_sub, ctx_sub)
                        if (Z_new - Z_sub).norm() < 1e-7:
                            break
                        Z_sub = Z_new
                J_z, J_A, _ = _compute_structural_jacobian(
                    lambda z, c: model.operator(z, c), Z_sub, ctx_sub
                )
                S = structural_sensitivity_matrix(
                    lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
                )
            else:
                S, _ = compute_explicit_sensitivity(model, X_sub, A_sub)

            S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
            if not edge_list or len(edge_list) < 5:
                continue

            aegis_scores = torch.norm(S_c, dim=0).cpu().numpy()

            grad_map = gradient_edge_attribution(model, X_sub, A_sub, is_ignn=is_ignn)
            grad_scores = []
            for (i, j) in edge_list:
                grad_scores.append(float(grad_map[i, j]))
            grad_scores = np.array(grad_scores)

            if is_ignn:
                bf_shifts = brute_force_ranking_generic(
                    model, X_sub, A_sub, edge_list,
                    is_ignn=True, ctx_sub=ctx_sub, Z_sub=Z_sub
                )
            else:
                bf_shifts = brute_force_ranking(model, X_sub, A_sub, edge_list)
            bf_scores = np.array(bf_shifts)

            tau_a, _ = kendalltau(aegis_scores, bf_scores)
            tau_g, _ = kendalltau(grad_scores, bf_scores)
            if not np.isnan(tau_a):
                tau_aegis_list.append(tau_a)
            if not np.isnan(tau_g):
                tau_grad_list.append(tau_g)

            top5_aegis = set(np.argsort(-aegis_scores)[:5])
            top5_grad = set(np.argsort(-grad_scores)[:5])
            top5_bf = set(np.argsort(-bf_scores)[:5])
            overlap_at5.append(len(top5_aegis & top5_bf) / 5)

        if tau_aegis_list:
            print(f"\n{model_name}:")
            print(f"  AEGIS tau (vs brute-force): {np.mean(tau_aegis_list):.3f} +/- {np.std(tau_aegis_list):.3f}")
            print(f"  Grad  tau (vs brute-force): {np.mean(tau_grad_list):.3f} +/- {np.std(tau_grad_list):.3f}")
            print(f"  AEGIS P@5 (vs brute-force): {np.mean(overlap_at5):.2f} +/- {np.std(overlap_at5):.2f}")


# ---------------------------------------------------------------
# Experiment 2: Degree-weighted S_c
# ---------------------------------------------------------------

def run_degree_weighted_sc(data, device):
    """Test if degree-weighting S_c columns improves ranking correlation."""
    print("\n" + "=" * 60)
    print("Experiment 2: Degree-Weighted S_c Ablation (10 seeds)")
    print("=" * 60)

    tau_uniform = []
    tau_degree = []
    tau_invdeg = []

    for seed in SEEDS:
        model, _ = train_model(ExplicitGCN, data, seed, device, n_layers=2)
        X = data["X"].to(device)
        A_hat = data["A_hat"].to(device)

        idx = extract_ego_subgraph(A_hat, max_nodes=50)
        A_sub = A_hat[idx][:, idx]
        X_sub = X[idx]

        S, _ = compute_explicit_sensitivity(model, X_sub, A_sub)
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        if not edge_list or len(edge_list) < 5:
            continue

        degrees = A_sub.sum(dim=1)
        edge_deg_weights = []
        for (i, j) in edge_list:
            edge_deg_weights.append(float(degrees[i] + degrees[j]) / 2.0)
        edge_deg_weights = torch.tensor(edge_deg_weights, device=device)

        scores_uniform = torch.norm(S_c, dim=0).cpu().numpy()
        scores_deg = (torch.norm(S_c, dim=0) * edge_deg_weights).cpu().numpy()
        scores_invdeg = (torch.norm(S_c, dim=0) / edge_deg_weights.clamp(min=1)).cpu().numpy()

        bf_shifts = brute_force_ranking(model, X_sub, A_sub, edge_list)
        bf_scores = np.array(bf_shifts)

        t_u, _ = kendalltau(scores_uniform, bf_scores)
        t_d, _ = kendalltau(scores_deg, bf_scores)
        t_i, _ = kendalltau(scores_invdeg, bf_scores)

        if not np.isnan(t_u):
            tau_uniform.append(t_u)
            tau_degree.append(t_d)
            tau_invdeg.append(t_i)

    print(f"\nGCN-2 on Cora (ranking vs brute-force removal):")
    print(f"  S_c uniform (default): tau = {np.mean(tau_uniform):.3f} +/- {np.std(tau_uniform):.3f}")
    print(f"  S_c * degree_weight:   tau = {np.mean(tau_degree):.3f} +/- {np.std(tau_degree):.3f}")
    print(f"  S_c / degree_weight:   tau = {np.mean(tau_invdeg):.3f} +/- {np.std(tau_invdeg):.3f}")


# ---------------------------------------------------------------
# Experiment 3: Scalability beyond N=200
# ---------------------------------------------------------------

def run_scalability_experiment(data, device):
    """Run AEGIS on larger subgraph sizes (N=300, 400, 500) to test scalability."""
    print("\n" + "=" * 60)
    print("Experiment 3: Scalability Beyond N=200 (10 seeds)")
    print("=" * 60)

    subgraph_sizes = [200, 300, 400, 500]

    for N in subgraph_sizes:
        tightness_list = []
        atk_adv_list = []
        time_list = []
        mem_list = []

        for seed in SEEDS:
            model, _ = train_model(ExplicitGCN, data, seed, device, n_layers=2)
            X = data["X"].to(device)
            A_hat = data["A_hat"].to(device)

            idx = extract_ego_subgraph(A_hat, max_nodes=N)
            actual_N = len(idx)
            if actual_N < N * 0.8:
                continue

            A_sub = A_hat[idx][:, idx]
            X_sub = X[idx]

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            S, Z_base = compute_explicit_sensitivity(model, X_sub, A_sub)
            S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

            elapsed = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

            if not edge_list:
                continue

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
                rand_w = torch.randn(len(edge_list), device=device)
                rand_w *= eps / rand_w.norm()
                rand_dA = torch.zeros_like(A_sub)
                for k, (i, j) in enumerate(edge_list):
                    rand_dA[i, j] = float(rand_w[k])
                    rand_dA[j, i] = float(rand_w[k])
                Z_rand = model.forward_hidden(X_sub, A_sub + rand_dA).reshape(-1)
            rand_shift = float((Z_rand - Z_base).norm())
            atk_adv = actual / rand_shift if rand_shift > 1e-12 else float("nan")

            if not np.isnan(tightness):
                tightness_list.append(tightness)
                atk_adv_list.append(atk_adv)
                time_list.append(elapsed)
                mem_list.append(peak_mem)

        if tightness_list:
            print(f"\nN={N} (actual nodes={actual_N}):")
            print(f"  Tightness: {np.mean(tightness_list):.3f} +/- {np.std(tightness_list):.3f}")
            print(f"  AtkAdv:    {np.mean(atk_adv_list):.1f} +/- {np.std(atk_adv_list):.1f}x")
            print(f"  Time:      {np.mean(time_list):.1f}s")
            print(f"  GPU Mem:   {np.mean(mem_list):.2f} GB")


# ---------------------------------------------------------------
# Experiment 4: Full-graph test accuracy for all models
# ---------------------------------------------------------------

def run_task_accuracy(data, device):
    """Report full-graph test accuracy (not subgraph) for all 7 architectures."""
    print("\n" + "=" * 60)
    print("Experiment 4: Full-Graph Test Accuracy (10 seeds)")
    print("=" * 60)

    models_cfg = [
        ('IGNN', IGNN, {}, True),
        ('GCN-2', ExplicitGCN, {'n_layers': 2}, False),
        ('GCN-4', ExplicitGCN, {'n_layers': 4}, False),
        ('GIN-2', ExplicitGIN, {'n_layers': 2}, False),
        ('GAT-2', ExplicitGAT, {'n_layers': 2}, False),
        ('SAGE-2', ExplicitGraphSAGE, {'n_layers': 2}, False),
        ('APPNP', ExplicitAPPNP, {}, False),
    ]

    print(f"\n{'Model':<10} {'Test Acc':<15}")
    print("-" * 30)

    for name, cls, kwargs, is_ignn in models_cfg:
        accs = []
        for seed in SEEDS:
            _, test_acc = train_model(cls, data, seed, device, is_ignn=is_ignn, **kwargs)
            accs.append(test_acc * 100)
        print(f"{name:<10} {np.mean(accs):.1f} +/- {np.std(accs):.1f}%")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_cora()
    print(f"Cora: {data['X'].shape[0]} nodes, {data['n_features']} features, {data['n_classes']} classes")

    run_task_accuracy(data, device)
    run_explainer_comparison(data, device)
    run_degree_weighted_sc(data, device)
    run_scalability_experiment(data, device)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
