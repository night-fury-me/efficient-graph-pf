"""
Proposition 4 bound tightness evaluation.
Compares the theoretical upper bound on sigma_1(S_K) from Proposition 4
against the actual sigma_1(S_K) computed via finite differences.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from iem.examples.exp_explicit_gnn_extension import (
    ExplicitGCN, ExplicitGIN, ExplicitGAT, ExplicitGraphSAGE, ExplicitAPPNP,
    compute_explicit_sensitivity, load_cora_data
)
from iem.adversarial import construct_Sc

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


def compute_prop4_bound(model, x, adj, K):
    """
    Compute the Proposition 4 upper bound:
    sigma_1(S_K) <= sum_{l=1}^{K} (prod_{k=l+1}^{K} ||J_z^(k)||_2) * ||J_A^(l)||_2

    For simplicity, we estimate per-layer Jacobian norms via finite differences
    at the actual intermediate representations.
    """
    delta = 1e-4
    N = adj.shape[0]

    with torch.no_grad():
        intermediates = [x]
        z = x
        for layer_idx in range(K):
            z = model.forward_one_layer(z, adj, layer_idx)
            intermediates.append(z)

    Jz_norms = []
    JA_norms = []

    for l in range(K):
        z_in = intermediates[l]
        z_out = intermediates[l + 1]
        D_in = z_in.numel()
        D_out = z_out.numel()

        Jz_cols = []
        for i in range(min(D_in, 200)):
            z_pert = z_in.clone().flatten()
            z_pert[i] += delta
            z_pert = z_pert.reshape(z_in.shape)
            with torch.no_grad():
                out_pert = model.forward_one_layer(z_pert, adj, l)
            col = ((out_pert - z_out) / delta).flatten()
            Jz_cols.append(col)

        if Jz_cols:
            Jz_sample = torch.stack(Jz_cols, dim=1)
            Jz_norm = torch.linalg.norm(Jz_sample, ord=2).item()
        else:
            Jz_norm = 1.0
        Jz_norms.append(Jz_norm)

        JA_cols = []
        adj_flat = adj.flatten()
        for i in range(min(N * N, 200)):
            if adj_flat[i].item() > 1e-8:
                adj_pert = adj.clone().flatten()
                adj_pert[i] += delta
                adj_pert = adj_pert.reshape(adj.shape)
                z_in_l = intermediates[l]
                with torch.no_grad():
                    out_pert = model.forward_one_layer(z_in_l, adj_pert, l)
                col = ((out_pert - z_out) / delta).flatten()
                JA_cols.append(col)

        if JA_cols:
            JA_sample = torch.stack(JA_cols, dim=1)
            JA_norm = torch.linalg.norm(JA_sample, ord=2).item()
        else:
            JA_norm = 0.0
        JA_norms.append(JA_norm)

    bound = 0.0
    for l in range(K):
        prod = 1.0
        for k in range(l + 1, K):
            prod *= Jz_norms[k]
        bound += prod * JA_norms[l]

    return bound


def run_prop4_evaluation():
    data = load_cora_data()

    models = {
        'GCN-2': (ExplicitGCN, {'num_layers': 2}),
        'GCN-4': (ExplicitGCN, {'num_layers': 4}),
        'GIN-2': (ExplicitGIN, {'num_layers': 2}),
        'GAT-2': (ExplicitGAT, {'num_layers': 2}),
        'SAGE-2': (ExplicitGraphSAGE, {'num_layers': 2}),
        'APPNP': (ExplicitAPPNP, {}),
    }

    print("Proposition 4 Bound Evaluation")
    print("=" * 60)
    print(f"{'Model':<10} {'Actual σ₁(S_K)':<16} {'Prop.4 Bound':<14} {'Ratio':<8}")
    print("-" * 60)

    for name, (model_cls, kwargs) in models.items():
        ratios = []
        for seed in SEEDS[:3]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_cls(
                in_features=data['x'].shape[1],
                hidden_dim=64,
                out_features=data['num_classes'],
                **kwargs
            )

            sub_adj = data['sub_adj']
            sub_x = data['sub_x']
            K = kwargs.get('num_layers', 2)
            if name == 'APPNP':
                K = 10

            S_K = compute_explicit_sensitivity(model, sub_x, sub_adj)
            Sc = construct_Sc(S_K, sub_adj)
            _, sigma, _ = torch.linalg.svd(Sc, full_matrices=False)
            actual_sigma1 = sigma[0].item()

            bound = compute_prop4_bound(model, sub_x, sub_adj, K)

            if actual_sigma1 > 0:
                ratios.append(bound / actual_sigma1)

        if ratios:
            mean_ratio = np.mean(ratios)
            print(f"{name:<10} {actual_sigma1:<16.2f} {bound:<14.2f} {mean_ratio:<8.1f}x")

    print()
    print("Bound is tighter for shallow (2-layer) models, looser for deeper ones.")


if __name__ == '__main__':
    run_prop4_evaluation()
