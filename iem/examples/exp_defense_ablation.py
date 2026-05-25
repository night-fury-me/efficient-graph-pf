"""
Defense-informed edge protection experiment.
Masks top-k vulnerable edges from the perturbation space
and measures reduction in SVD attack damage.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from iem.examples.ignn_cora import load_cora, build_ignn, extract_ego_subgraph
from iem.ift import compute_jacobians, solve_sensitivity
from iem.adversarial import (
    construct_Sc, optimal_structural_attack, per_node_robust_radius
)

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
TOP_K_VALUES = [5, 10]
EPS = 0.01


def run_defense_ablation():
    results = {k: {'aegis_reduction': [], 'random_reduction': []} for k in TOP_K_VALUES}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = load_cora()
        model = build_ignn(data, seed=seed)

        sub_adj, sub_x, sub_y, node_map = extract_ego_subgraph(
            data, center_node=seed % data.num_nodes, num_nodes=50
        )

        z_star = model.forward_to_equilibrium(sub_x, sub_adj)
        Jz, JA = compute_jacobians(model, z_star, sub_adj)
        S = solve_sensitivity(Jz, JA)
        Sc = construct_Sc(S, sub_adj)

        U, sigma, Vt = torch.linalg.svd(Sc, full_matrices=False)
        base_damage = sigma[0].item() * EPS

        edge_vulns = torch.norm(Sc, dim=0)
        sorted_edges = torch.argsort(edge_vulns, descending=True)

        for k in TOP_K_VALUES:
            top_k_mask = sorted_edges[:k]
            Sc_protected = Sc.clone()
            Sc_protected[:, top_k_mask] = 0.0

            _, sigma_prot, _ = torch.linalg.svd(Sc_protected, full_matrices=False)
            protected_damage = sigma_prot[0].item() * EPS
            aegis_reduction = 1.0 - (protected_damage / base_damage)
            results[k]['aegis_reduction'].append(aegis_reduction)

            random_reductions = []
            for _ in range(10):
                rand_mask = torch.randperm(Sc.shape[1])[:k]
                Sc_rand = Sc.clone()
                Sc_rand[:, rand_mask] = 0.0
                _, sigma_rand, _ = torch.linalg.svd(Sc_rand, full_matrices=False)
                rand_damage = sigma_rand[0].item() * EPS
                random_reductions.append(1.0 - (rand_damage / base_damage))
            results[k]['random_reduction'].append(np.mean(random_reductions))

    print("Defense-Informed Edge Protection Results")
    print("=" * 50)
    for k in TOP_K_VALUES:
        aegis_mean = np.mean(results[k]['aegis_reduction']) * 100
        aegis_std = np.std(results[k]['aegis_reduction']) * 100
        rand_mean = np.mean(results[k]['random_reduction']) * 100
        rand_std = np.std(results[k]['random_reduction']) * 100
        print(f"Top-{k} masking:")
        print(f"  AEGIS-guided: {aegis_mean:.0f} +/- {aegis_std:.0f}% reduction")
        print(f"  Random:       {rand_mean:.0f} +/- {rand_std:.0f}% reduction")
        print()


if __name__ == '__main__':
    run_defense_ablation()
