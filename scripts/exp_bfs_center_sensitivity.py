"""BFS Center Sensitivity Analysis for AEGIS paper.

Reviewer concern: does using highest-degree node as BFS center bias
vulnerability rankings?

For each seed, trains IGNN on Cora, extracts 3 different 50-node BFS
subgraphs (highest-degree, random, median-degree centers), computes S_c
edge vulnerability rankings, and compares them via Kendall tau on
overlapping edges. Also computes tau vs discrete ground truth (greedy
brute-force edge removal).

Output: results/bfs_center_sensitivity.txt

Usage:
    .venv/bin/python scripts/exp_bfs_center_sensitivity.py
"""

from __future__ import annotations

import gc
import random
import sys
import time
from collections import deque
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
    greedy_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora

SEEDS = [42, 137, 271, 314, 1729]
MAX_NODES = 50
TRAIN_EPOCHS = 50
OUTFILE = Path(__file__).resolve().parents[1] / "results" / "bfs_center_sensitivity.txt"


# ---- Custom BFS extraction with specified center ----

def bfs_subgraph(A_hat: torch.Tensor, center: int, max_nodes: int = 50) -> torch.Tensor:
    """BFS-based subgraph extraction from a specified center node."""
    visited = [center]
    seen = {center}
    queue = deque([center])
    while len(visited) < max_nodes and queue:
        node = queue.popleft()
        for n in (A_hat[node] > 0).nonzero(as_tuple=True)[0].tolist():
            if n not in seen and len(visited) < max_nodes:
                seen.add(n)
                visited.append(n)
                queue.append(n)
    return torch.tensor(sorted(visited), device=A_hat.device)


def pick_centers(A_hat: torch.Tensor, rng: random.Random) -> dict:
    """Pick three center nodes: highest-degree, random, median-degree."""
    degrees = A_hat.sum(dim=1)
    N = A_hat.shape[0]

    # Highest-degree (the current default)
    highest = int(degrees.argmax().item())

    # Random node (seeded)
    random_center = rng.randint(0, N - 1)

    # Median-degree node
    sorted_idx = degrees.argsort()
    median_center = int(sorted_idx[N // 2].item())

    return {
        "highest_degree": highest,
        "random": random_center,
        "median_degree": median_center,
    }


def train_ignn(data, device, seed):
    """Train IGNN with given seed, return model + equilibrium."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for ep in range(1, TRAIN_EPOCHS + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
    test_acc = float((logits.argmax(1)[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    return model, Z_star, ctx, A_hat, test_acc


def compute_vulnerability_ranking(model, Z_star, ctx, A_hat, subgraph_idx):
    """Compute S_c-based vulnerability scores for edges in subgraph.

    Returns:
        edge_scores: dict mapping (i, j) -> vulnerability score (local indices)
        edge_list: list of (i, j) local-index edges
    """
    S = len(subgraph_idx)
    A_sub = A_hat[subgraph_idx][:, subgraph_idx]
    X_proj_sub = ctx["X_proj"][subgraph_idx]
    Z_star_sub = Z_star[subgraph_idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    def F_ignn(z, c=ctx_sub):
        return model.operator(z, c)

    # Structural sensitivity matrix
    S_mat = structural_sensitivity_matrix(F_ignn, Z_star_sub, ctx_sub)
    S_c, edge_list = constrained_sensitivity_matrix(S_mat, A_sub)

    # Per-edge vulnerability score = ||column of S_c||_2
    edge_scores = {}
    for k, (i, j) in enumerate(edge_list):
        score = float(S_c[:, k].norm())
        edge_scores[(i, j)] = score

    del S_mat, S_c
    return edge_scores, edge_list


def compute_discrete_ranking(model, Z_star, ctx, A_hat, subgraph_idx):
    """Greedy brute-force edge removal on subgraph (discrete ground truth)."""
    A_sub = A_hat[subgraph_idx][:, subgraph_idx]
    X_proj_sub = ctx["X_proj"][subgraph_idx]
    Z_star_sub = Z_star[subgraph_idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    results = greedy_structural_attack(model, Z_star_sub, ctx_sub)
    # results: list of (i, j, damage) sorted by descending damage
    edge_scores = {}
    for i, j, damage in results:
        edge_scores[(i, j)] = damage
    return edge_scores


def kendall_tau_on_overlap(scores_a: dict, scores_b: dict):
    """Compute Kendall tau on overlapping edges between two rankings."""
    overlap = set(scores_a.keys()) & set(scores_b.keys())
    if len(overlap) < 3:
        return float("nan"), 0
    edges = sorted(overlap)
    a = np.array([scores_a[e] for e in edges])
    b = np.array([scores_b[e] for e in edges])
    tau, pval = kendalltau(a, b)
    return tau, len(overlap)


def remap_to_local(edge_scores_global_sub, subgraph_idx):
    """Edge scores use local indices within a subgraph. No remapping needed
    if computed on the same subgraph. This is for cross-subgraph comparison:
    we map local indices back to global, then compare."""
    idx_list = subgraph_idx.tolist()
    global_scores = {}
    for (li, lj), score in edge_scores_global_sub.items():
        gi, gj = idx_list[li], idx_list[lj]
        if gi > gj:
            gi, gj = gj, gi
        global_scores[(gi, gj)] = score
    return global_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/cora")

    print("=" * 70)
    print("BFS Center Sensitivity Analysis — AEGIS Paper")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Subgraph size: {MAX_NODES} nodes")
    print()

    _download_cora(data_dir)
    data = _load_cora(data_dir)
    print(f"Cora loaded: N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")
    print()

    # Accumulators across seeds
    all_pairwise_tau = {"highest_vs_random": [], "highest_vs_median": [], "random_vs_median": []}
    all_discrete_tau = {"highest_degree": [], "random": [], "median_degree": []}
    all_overlaps = {"highest_vs_random": [], "highest_vs_median": [], "random_vs_median": []}

    lines = []  # for output file

    for seed in SEEDS:
        t0 = time.time()
        print(f"--- Seed {seed} ---")

        rng = random.Random(seed)
        model, Z_star, ctx, A_hat, test_acc = train_ignn(data, device, seed)
        print(f"  Test accuracy: {test_acc:.3f}")

        centers = pick_centers(A_hat, rng)
        print(f"  Centers: highest_degree={centers['highest_degree']}, "
              f"random={centers['random']}, median_degree={centers['median_degree']}")

        # Extract subgraphs and compute vulnerability rankings
        subgraphs = {}
        continuous_scores = {}  # global-index edge -> score
        discrete_scores = {}   # global-index edge -> score
        strategies = ["highest_degree", "random", "median_degree"]

        for strategy in strategies:
            center = centers[strategy]
            sub_idx = bfs_subgraph(A_hat, center, MAX_NODES)
            subgraphs[strategy] = sub_idx
            n_nodes = len(sub_idx)

            # Count edges in subgraph
            A_sub = A_hat[sub_idx][:, sub_idx]
            n_edges = int((A_sub.abs() > 1e-10).sum().item() - n_nodes) // 2  # exclude self-loops
            print(f"  {strategy}: {n_nodes} nodes, {n_edges} edges (center={center})")

            # Continuous (S_c) ranking
            c_scores, _ = compute_vulnerability_ranking(model, Z_star, ctx, A_hat, sub_idx)
            continuous_scores[strategy] = remap_to_local(c_scores, sub_idx)

            # Discrete (greedy) ground truth
            d_scores = compute_discrete_ranking(model, Z_star, ctx, A_hat, sub_idx)
            discrete_scores[strategy] = remap_to_local(d_scores, sub_idx)

            gc.collect()

        # Pairwise Kendall tau between continuous rankings from different centers
        pairs = [
            ("highest_vs_random", "highest_degree", "random"),
            ("highest_vs_median", "highest_degree", "median_degree"),
            ("random_vs_median", "random", "median_degree"),
        ]
        print("  Pairwise tau (continuous rankings, overlapping edges):")
        for label, s1, s2 in pairs:
            tau, n_overlap = kendall_tau_on_overlap(
                continuous_scores[s1], continuous_scores[s2]
            )
            all_pairwise_tau[label].append(tau)
            all_overlaps[label].append(n_overlap)
            print(f"    {label}: tau={tau:.4f}, overlap={n_overlap} edges")

        # Tau vs discrete ground truth for each center strategy
        print("  Tau vs discrete ground truth:")
        for strategy in strategies:
            tau, n_overlap = kendall_tau_on_overlap(
                continuous_scores[strategy], discrete_scores[strategy]
            )
            all_discrete_tau[strategy].append(tau)
            print(f"    {strategy}: tau={tau:.4f}, overlap={n_overlap} edges")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        print()

        del model, Z_star, ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_lines = []

    def report(msg):
        print(msg)
        summary_lines.append(msg)

    report("BFS Center Sensitivity Analysis — AEGIS Paper")
    report(f"Seeds: {SEEDS}, Subgraph size: {MAX_NODES} nodes")
    report("")

    report("1. Pairwise Kendall tau between continuous vulnerability rankings")
    report("   (computed on overlapping edges across different BFS centers):")
    report("")
    for label in all_pairwise_tau:
        vals = [v for v in all_pairwise_tau[label] if not np.isnan(v)]
        overlaps = all_overlaps[label]
        if vals:
            mean_tau = np.mean(vals)
            std_tau = np.std(vals)
            mean_overlap = np.mean(overlaps)
            report(f"   {label}: tau = {mean_tau:.4f} +/- {std_tau:.4f}  "
                   f"(mean overlap: {mean_overlap:.0f} edges, n={len(vals)} seeds)")
        else:
            report(f"   {label}: insufficient overlap across all seeds")

    report("")
    report("2. Kendall tau vs discrete ground truth (greedy edge removal)")
    report("   for each center strategy:")
    report("")
    for strategy in strategies:
        vals = [v for v in all_discrete_tau[strategy] if not np.isnan(v)]
        if vals:
            mean_tau = np.mean(vals)
            std_tau = np.std(vals)
            report(f"   {strategy}: tau = {mean_tau:.4f} +/- {std_tau:.4f}  (n={len(vals)} seeds)")
        else:
            report(f"   {strategy}: insufficient data")

    report("")

    # Interpretation
    pairwise_means = {}
    for label in all_pairwise_tau:
        vals = [v for v in all_pairwise_tau[label] if not np.isnan(v)]
        pairwise_means[label] = np.mean(vals) if vals else float("nan")

    discrete_means = {}
    for strategy in strategies:
        vals = [v for v in all_discrete_tau[strategy] if not np.isnan(v)]
        discrete_means[strategy] = np.mean(vals) if vals else float("nan")

    report("3. Interpretation:")
    report("")

    # Check if pairwise tau values are high (> 0.6 considered strong agreement)
    high_agreement = all(v > 0.6 for v in pairwise_means.values() if not np.isnan(v))
    moderate_agreement = all(v > 0.4 for v in pairwise_means.values() if not np.isnan(v))

    if high_agreement:
        report("   FINDING: Strong agreement (tau > 0.6) between vulnerability rankings")
        report("   from different BFS centers. Center choice does NOT significantly bias")
        report("   the vulnerability analysis.")
    elif moderate_agreement:
        report("   FINDING: Moderate agreement (tau > 0.4) between vulnerability rankings")
        report("   from different BFS centers. Center choice has limited effect on")
        report("   vulnerability analysis.")
    else:
        report("   FINDING: Weak agreement between rankings from different BFS centers.")
        report("   Center choice may affect vulnerability analysis — results should be")
        report("   interpreted with caution.")

    report("")

    # Check discrete tau consistency
    disc_vals = [v for v in discrete_means.values() if not np.isnan(v)]
    if disc_vals:
        disc_range = max(disc_vals) - min(disc_vals)
        best = max(discrete_means, key=lambda k: discrete_means[k])
        report(f"   Discrete tau range across strategies: {disc_range:.4f}")
        report(f"   Best center strategy (vs ground truth): {best} "
               f"(tau = {discrete_means[best]:.4f})")
        if disc_range < 0.10:
            report("   All center strategies yield comparable fidelity to ground truth.")
        else:
            report(f"   Notable variation: {best} outperforms others by {disc_range:.4f} tau.")

    report("")
    report("4. Per-seed detail:")
    report("")
    for i, seed in enumerate(SEEDS):
        report(f"   Seed {seed}:")
        for label in all_pairwise_tau:
            v = all_pairwise_tau[label][i]
            o = all_overlaps[label][i]
            report(f"     {label}: tau={v:.4f}, overlap={o}")
        for strategy in strategies:
            v = all_discrete_tau[strategy][i]
            report(f"     {strategy} vs discrete: tau={v:.4f}")
        report("")

    # Save to file
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTFILE, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"Results saved to {OUTFILE}")


if __name__ == "__main__":
    main()
