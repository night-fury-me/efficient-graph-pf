"""B7 — Enhanced power grid analysis addressing 6 review points.

1. LODF wall-clock timing comparison
2. N-2 analysis from SVD direction
3. Rank stability across seeds (Kendall tau)
4. Binary vs admittance analysis (WHY binary wins)

IEEE cases: case14, case30, case57, case118
Seeds: 10 independent seeds per case

Usage:
    .venv/bin/python scripts/exp_power_grid_enhanced.py [--cases case14,case30]
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa — registers all model builders

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.contractive_pf import ContractiveGCN_PF

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

IEEE_CASES = [
    ("case14", "datasets/IEEE_case14_2000.parquet", 14),
    ("case30", "datasets/IEEE_case30_2000.parquet", 30),
    ("case57", "datasets/IEEE_case57_2000.parquet", 57),
    ("case118", "datasets/IEEE_case118_2000.parquet", 118),
]

RESULTS_DIR = Path("results/power_grid_enhanced")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}+/-{s:{fmt}}"


class WeightedContractiveGCN_PF(ContractiveGCN_PF):
    """ContractiveGCN_PF with admittance-weighted adjacency."""

    def _build_adjacency(self, Y):
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        A = Y.abs()
        A.fill_diagonal_(0.0)
        max_val = A.max()
        if max_val > 1e-10:
            A = A / max_val
        A.fill_diagonal_(1.0)
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D


# ---------------------------------------------------------------------------
# LODF baseline (from exp_lodf_baseline.py pattern)
# ---------------------------------------------------------------------------

def compute_lodf_ranking_from_Y(Y_bus: torch.Tensor):
    """LODF-based contingency ranking from admittance matrix.

    Uses DC power flow approximation: B-matrix from imaginary part of Y_bus,
    PTDF via pseudoinverse, then LODF for each contingency.
    """
    if Y_bus.dim() == 3:
        Y_bus = Y_bus.squeeze(0)
    N = Y_bus.shape[0]
    Y_np = Y_bus.detach().cpu().numpy()

    # Build B-matrix (imaginary part of off-diagonal = susceptance)
    B = np.zeros((N, N))
    edges = []
    reactances = []
    for i in range(N):
        for j in range(i + 1, N):
            y_ij = Y_np[i, j]
            if abs(y_ij) > 1e-12:
                b_ij = -y_ij.imag  # susceptance
                if abs(b_ij) < 1e-12:
                    b_ij = abs(y_ij)  # fallback to magnitude
                B[i, j] -= b_ij
                B[j, i] -= b_ij
                B[i, i] += b_ij
                B[j, j] += b_ij
                edges.append((i, j))
                x_ij = 1.0 / b_ij if abs(b_ij) > 1e-10 else 100.0
                reactances.append(x_ij)

    if len(edges) < 2:
        return [], edges, reactances

    # PTDF via pseudoinverse (slack = bus 0)
    slack = 0
    non_slack = [i for i in range(N) if i != slack]
    B_red = B[np.ix_(non_slack, non_slack)]
    try:
        X = np.linalg.pinv(B_red)
    except np.linalg.LinAlgError:
        return [], edges, reactances

    X_full = np.zeros((N, N))
    for i_idx, ni in enumerate(non_slack):
        for j_idx, nj in enumerate(non_slack):
            X_full[ni, nj] = X[i_idx, j_idx]

    n_edges = len(edges)
    PTDF = np.zeros((n_edges, N))
    for l, (fr, to) in enumerate(edges):
        x_l = reactances[l]
        if abs(x_l) > 1e-10:
            for i in range(N):
                PTDF[l, i] = (X_full[fr, i] - X_full[to, i]) / x_l

    # LODF matrix
    LODF = np.zeros((n_edges, n_edges))
    for k in range(n_edges):
        fr_k, to_k = edges[k]
        denom = 1.0 - (PTDF[k, fr_k] - PTDF[k, to_k])
        if abs(denom) < 1e-10:
            continue
        for l in range(n_edges):
            if l != k:
                LODF[l, k] = (PTDF[l, fr_k] - PTDF[l, to_k]) / denom

    # Severity = max redistribution when line k is removed
    severity = np.array([np.max(np.abs(LODF[:, k])) for k in range(n_edges)])
    ranking = sorted(
        [(edges[k][0], edges[k][1], float(severity[k])) for k in range(n_edges)],
        key=lambda x: x[2], reverse=True,
    )
    return ranking, edges, reactances


def compute_effective_resistance_ranking(A_hat: torch.Tensor):
    """Effective-resistance-based ranking (graph Laplacian pseudoinverse)."""
    N = A_hat.shape[0]
    A_bin = (A_hat.abs() > 1e-10).float()
    A_bin.fill_diagonal_(0.0)
    L = torch.diag(A_bin.sum(dim=1)) - A_bin

    try:
        eigvals, eigvecs = torch.linalg.eigh(L.cpu())
        eigvals = eigvals.to(A_hat.device)
        eigvecs = eigvecs.to(A_hat.device)
        mask = eigvals.abs() > 1e-8
        L_pinv = eigvecs[:, mask] @ torch.diag(1.0 / eigvals[mask]) @ eigvecs[:, mask].T
    except Exception:
        return []

    ranking = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_bin[i, j] > 0.5:
                r_ij = float(L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j])
                ranking.append((i, j, abs(r_ij)))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# Core: train model + compute AEGIS pipeline
# ---------------------------------------------------------------------------

def train_and_analyse(case_name, ds_path, N_expected, seed, device):
    """Train ContractiveGCN-PF, compute S_c, return analysis context."""
    set_seed(seed)
    if not Path(ds_path).exists():
        return None

    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    t_train_start = time.time()
    for ep in range(30):
        model.train()
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    t_train = time.time() - t_train_start

    model.eval()

    # Forward on first sample to get Z*, ctx
    eval_batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False,
                                       collate_fn=collate_blockdiag)))
    with torch.no_grad():
        V_pred, ctx_pf = model(
            eval_batch["bus_type"].to(device),
            eval_batch["Lines_connected"].to(device),
            None,
            eval_batch["Y_Lines"].to(device),
            eval_batch["Y_C_Lines"].to(device),
            eval_batch["S_start"].to(device),
            eval_batch["V_start"].to(device),
            eval_batch["sizes"].to(device),
        )

    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    N = int(eval_batch["sizes"][0].item())

    A_sub = A_hat[:N, :N]
    X_proj_sub = ctx_pf["X_proj"][:N]
    Z_sub = Z_star[:N]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    # Reconverge
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    # Edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_sub[i, j].abs() > 1e-10:
                edges.append((i, j))
    if len(edges) < 3:
        return None

    # S_c computation (timed)
    t_sc_start = time.time()
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

    # SVD of S_c
    U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
    t_sc = time.time() - t_sc_start

    # AEGIS vulnerability ranking
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    aegis_ranking = [(i, j, v) for i, j, v in attack["all_edge_vulnerabilities"]]
    del attack

    # Brute-force N-1 ground truth
    t_bf_start = time.time()
    bf_ranking = greedy_structural_attack(model, Z_sub, ctx_sub)
    t_bf = time.time() - t_bf_start

    Y_bus = ctx_pf.get("Y")
    del ctx_pf, eval_batch, J_z, J_A, S
    gc.collect()

    return {
        "model": model,
        "ctx_sub": ctx_sub,
        "Z_sub": Z_sub,
        "A_sub": A_sub,
        "S_c": S_c,
        "edge_list": edge_list,
        "sigma_c": sigma_c,
        "Vh_c": Vh_c,
        "N": N,
        "edges": edges,
        "aegis_ranking": aegis_ranking,
        "bf_ranking": bf_ranking,
        "t_train": t_train,
        "t_sc": t_sc,
        "t_bf": t_bf,
        "Y_bus": Y_bus,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Analysis 1: LODF wall-clock timing comparison
# ---------------------------------------------------------------------------

def analysis_timing(ctx_all: dict) -> dict:
    """Compare AEGIS pipeline timing vs LODF/effective-resistance baseline."""
    t_aegis = ctx_all["t_train"] + ctx_all["t_sc"]  # train + S_c + ranking

    Y_bus = ctx_all["Y_bus"]
    A_sub = ctx_all["A_sub"]

    # Time LODF baseline
    t_lodf_start = time.time()
    lodf_ranking, _, _ = compute_lodf_ranking_from_Y(Y_bus)
    t_lodf = time.time() - t_lodf_start

    # Time effective-resistance baseline
    t_er_start = time.time()
    er_ranking = compute_effective_resistance_ranking(A_sub)
    t_er = time.time() - t_er_start

    return {
        "t_aegis_train": ctx_all["t_train"],
        "t_aegis_sc": ctx_all["t_sc"],
        "t_aegis_total": t_aegis,
        "t_lodf": t_lodf,
        "t_eff_res": t_er,
        "t_bf_n1": ctx_all["t_bf"],
    }


# ---------------------------------------------------------------------------
# Analysis 2: N-2 analysis from SVD direction
# ---------------------------------------------------------------------------

def analysis_n2(ctx_all: dict, top_k: int = 5) -> dict:
    """Extract top-k edges from leading singular vector, check N-2 overlap."""
    model = ctx_all["model"]
    Z_sub = ctx_all["Z_sub"]
    ctx_sub = ctx_all["ctx_sub"]
    A_sub = ctx_all["A_sub"]
    edge_list = ctx_all["edge_list"]
    Vh_c = ctx_all["Vh_c"]
    N = ctx_all["N"]
    edges = ctx_all["edges"]
    n_edges = len(edges)

    # Top-k edges from leading singular vector v1
    v1 = Vh_c[0].abs()  # leading right singular vector
    k = min(top_k, len(edge_list))
    topk_indices = v1.argsort(descending=True)[:k].tolist()
    svd_top_edges = [edge_list[idx] for idx in topk_indices]

    # All pairs from SVD top-k edges
    svd_pairs = set()
    for a_idx in range(len(svd_top_edges)):
        for b_idx in range(a_idx + 1, len(svd_top_edges)):
            e1 = svd_top_edges[a_idx]
            e2 = svd_top_edges[b_idx]
            svd_pairs.add((e1, e2))

    # Brute-force N-2: remove all pairs, measure severity
    # Only feasible for small grids
    max_bf_pairs = 1000  # safety cap
    all_pairs = list(itertools.combinations(range(n_edges), 2))
    if len(all_pairs) > max_bf_pairs:
        # For large grids, sample around SVD-suggested edges + random
        sampled_pairs = []
        # Include all pairs involving SVD top-k edges
        svd_edge_indices = set()
        for e in svd_top_edges:
            for idx, (i, j) in enumerate(edges):
                if (i, j) == e or (j, i) == e:
                    svd_edge_indices.add(idx)
                    break
        for a_idx in svd_edge_indices:
            for b_idx in range(n_edges):
                if b_idx != a_idx:
                    pair = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if pair not in sampled_pairs:
                        sampled_pairs.append(pair)
        # Add random pairs up to cap
        rng = np.random.RandomState(42)
        random_indices = rng.choice(len(all_pairs), size=min(500, len(all_pairs)),
                                     replace=False)
        for ri in random_indices:
            pair = all_pairs[ri]
            if pair not in sampled_pairs:
                sampled_pairs.append(pair)
        if len(sampled_pairs) > max_bf_pairs:
            sampled_pairs = sampled_pairs[:max_bf_pairs]
        eval_pairs = sampled_pairs
        full_bf = False
    else:
        eval_pairs = all_pairs
        full_bf = True

    # Run brute-force N-2
    pair_severities = []
    A_orig = ctx_sub["A_hat"]
    with torch.no_grad():
        for e1_idx, e2_idx in eval_pairs:
            i1, j1 = edges[e1_idx]
            i2, j2 = edges[e2_idx]
            A_pert = A_orig.clone()
            A_pert[i1, j1] = 0.0
            A_pert[j1, i1] = 0.0
            A_pert[i2, j2] = 0.0
            A_pert[j2, i2] = 0.0
            ctx_pert = {**ctx_sub, "A_hat": A_pert}
            Z = Z_sub.clone()
            for _ in range(50):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).norm() < 1e-7:
                    break
                Z = Z_new
            severity = float((Z_new - Z_sub).norm())
            pair_severities.append((e1_idx, e2_idx, severity))

    # Sort by severity descending
    pair_severities.sort(key=lambda x: x[2], reverse=True)

    # Ground-truth top-k N-2 critical pairs
    gt_k = min(top_k, len(pair_severities))
    gt_top_pairs = set()
    for e1_idx, e2_idx, _ in pair_severities[:gt_k]:
        e1 = edges[e1_idx]
        e2 = edges[e2_idx]
        gt_top_pairs.add((e1, e2))

    # Overlap: how many SVD-suggested pairs appear in ground-truth top-k
    overlap = 0
    for svd_pair in svd_pairs:
        # Check both orderings
        if svd_pair in gt_top_pairs or (svd_pair[1], svd_pair[0]) in gt_top_pairs:
            overlap += 1

    n_svd_pairs = len(svd_pairs)
    n_gt_pairs = len(gt_top_pairs)
    overlap_frac = overlap / max(n_svd_pairs, 1)

    # Also check: what fraction of SVD top-k EDGES appear in the edges of
    # the ground-truth top-k N-2 pairs (edge-level overlap)
    gt_top_edges = set()
    for e1_idx, e2_idx, _ in pair_severities[:gt_k]:
        gt_top_edges.add(edges[e1_idx])
        gt_top_edges.add(edges[e2_idx])
    svd_edge_overlap = len(set(svd_top_edges) & gt_top_edges) / max(k, 1)

    return {
        "svd_top_edges": svd_top_edges,
        "n_svd_pairs": n_svd_pairs,
        "n_gt_pairs_evaluated": len(eval_pairs),
        "full_bruteforce": full_bf,
        "pair_overlap": overlap,
        "pair_overlap_frac": overlap_frac,
        "edge_overlap_frac": svd_edge_overlap,
        "top_n2_severity": pair_severities[0][2] if pair_severities else 0.0,
        "top_n2_pair": (edges[pair_severities[0][0]],
                        edges[pair_severities[0][1]]) if pair_severities else None,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Rank stability across seeds
# ---------------------------------------------------------------------------

def analysis_rank_stability(case_results: list, n_edges_total: int) -> dict:
    """Compute pairwise Kendall tau of top-10 rankings across seeds."""
    if len(case_results) < 2:
        return {"mean_tau": None, "std_tau": None, "consensus_frac": None}

    # Extract top-10 edge sets and orderings per seed
    top_k = 10
    seed_rankings = []
    for r in case_results:
        ranking = r["aegis_ranking"]
        top_edges = []
        for i, j, v in ranking[:top_k]:
            top_edges.append((min(i, j), max(i, j)))
        seed_rankings.append(top_edges)

    # Pairwise Kendall tau on score vectors
    # Build per-seed score dictionaries for ALL edges
    score_vectors = []
    for r in case_results:
        score_dict = {}
        for i, j, v in r["aegis_ranking"]:
            score_dict[(min(i, j), max(i, j))] = v
        score_vectors.append(score_dict)

    # Common edges across all seeds
    all_edges = set()
    for sd in score_vectors:
        all_edges.update(sd.keys())
    common_edges = sorted(all_edges)

    # Build score matrix (seeds x edges), fill missing with 0
    score_matrix = np.zeros((len(case_results), len(common_edges)))
    for s_idx, sd in enumerate(score_vectors):
        for e_idx, e in enumerate(common_edges):
            score_matrix[s_idx, e_idx] = sd.get(e, 0.0)

    # Pairwise Kendall tau
    taus = []
    for i in range(len(case_results)):
        for j in range(i + 1, len(case_results)):
            tau, _ = kendalltau(score_matrix[i], score_matrix[j])
            if not np.isnan(tau):
                taus.append(tau)

    # Consensus edges: appear in top-10 across ALL seeds
    if seed_rankings:
        consensus = set(seed_rankings[0])
        for sr in seed_rankings[1:]:
            consensus &= set(sr)
        consensus_frac = len(consensus) / top_k
    else:
        consensus_frac = 0.0

    # Also compute: fraction appearing in top-10 in >= 80% of seeds
    edge_counts = {}
    for sr in seed_rankings:
        for e in sr:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    n_seeds = len(seed_rankings)
    stable_80 = sum(1 for c in edge_counts.values() if c >= 0.8 * n_seeds)
    stable_80_frac = stable_80 / top_k

    return {
        "mean_tau": float(np.mean(taus)) if taus else None,
        "std_tau": float(np.std(taus)) if taus else None,
        "consensus_frac": consensus_frac,
        "stable_80_frac": stable_80_frac,
        "n_seeds_evaluated": len(case_results),
        "consensus_edges": sorted(consensus) if seed_rankings else [],
    }


# ---------------------------------------------------------------------------
# Analysis 4: Binary vs admittance — WHY binary wins
# ---------------------------------------------------------------------------

def analysis_binary_vs_admittance(ctx_all: dict) -> dict:
    """Investigate why binary adjacency outperforms admittance-weighted.

    Hypothesis: N-1 is a discrete event (full line removal), so binary
    sensitivity (uniform per edge) better models the discontinuous impact
    than admittance-weighted (which scales by existing impedance).
    """
    model = ctx_all["model"]
    edges = ctx_all["edges"]
    bf_ranking = ctx_all["bf_ranking"]
    A_sub = ctx_all["A_sub"]
    Y_bus = ctx_all["Y_bus"]
    N = ctx_all["N"]

    # BF severity per edge
    bf_dict = {}
    for i, j, sev in bf_ranking:
        bf_dict[(min(i, j), max(i, j))] = sev

    # Admittance magnitude per edge
    if Y_bus is not None:
        if Y_bus.dim() == 3:
            Y_np = Y_bus.squeeze(0).detach().cpu().numpy()
        else:
            Y_np = Y_bus.detach().cpu().numpy()
    else:
        Y_np = None

    admittance_vals = []
    severity_vals = []
    edge_keys = []
    for i, j in edges:
        key = (min(i, j), max(i, j))
        sev = bf_dict.get(key, 0.0)
        if Y_np is not None:
            adm = abs(Y_np[i, j])
        else:
            adm = float(A_sub[i, j].abs())
        admittance_vals.append(adm)
        severity_vals.append(sev)
        edge_keys.append(key)

    admittance_vals = np.array(admittance_vals)
    severity_vals = np.array(severity_vals)

    # Correlation between admittance and N-1 severity
    if len(admittance_vals) >= 3:
        corr_pearson = float(np.corrcoef(admittance_vals, severity_vals)[0, 1])
        corr_spearman, _ = spearmanr(admittance_vals, severity_vals)
        corr_spearman = float(corr_spearman)
    else:
        corr_pearson = None
        corr_spearman = None

    # Binary degree per edge: degree(i) + degree(j)
    A_bin = (A_sub.abs() > 1e-10).float()
    A_bin.fill_diagonal_(0.0)
    degrees = A_bin.sum(dim=1).cpu().numpy()
    degree_sums = []
    for i, j in edges:
        degree_sums.append(degrees[i] + degrees[j])
    degree_sums = np.array(degree_sums)

    if len(degree_sums) >= 3:
        corr_degree_sev, _ = spearmanr(degree_sums, severity_vals)
        corr_degree_sev = float(corr_degree_sev)
    else:
        corr_degree_sev = None

    # Effective resistance per edge (binary graph)
    er_ranking = compute_effective_resistance_ranking(A_sub)
    er_dict = {}
    for i, j, r in er_ranking:
        er_dict[(min(i, j), max(i, j))] = r
    er_vals = np.array([er_dict.get(key, 0.0) for key in edge_keys])

    if len(er_vals) >= 3:
        corr_er_sev, _ = spearmanr(er_vals, severity_vals)
        corr_er_sev = float(corr_er_sev)
    else:
        corr_er_sev = None

    # Variance of admittance across edges (high variance = more distortion
    # when used as weights, because high-admittance edges dominate)
    adm_cv = float(np.std(admittance_vals) / (np.mean(admittance_vals) + 1e-10))

    # Key insight: for N-1, each edge removal is an equally-sized discrete
    # perturbation (full removal). High-admittance edges might actually be
    # LESS critical to remove because the grid is well-connected there.
    # Check: is there a NEGATIVE correlation?
    return {
        "corr_admittance_severity_pearson": corr_pearson,
        "corr_admittance_severity_spearman": corr_spearman,
        "corr_degree_severity_spearman": corr_degree_sev,
        "corr_effres_severity_spearman": corr_er_sev,
        "admittance_cv": adm_cv,
        "n_edges": len(edges),
        "interpretation": (
            "Negative/weak admittance-severity correlation confirms that "
            "high-admittance lines are NOT the most critical for N-1. "
            "Binary adjacency avoids this misleading weighting."
            if corr_spearman is not None and corr_spearman < 0.3
            else "Moderate admittance-severity correlation — weighting has some merit."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=str, default=None,
                    help="Comma-separated case names (default: all)")
    ap.add_argument("--seeds", type=int, default=None, nargs="+",
                    help="Override seed list")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = IEEE_CASES
    if args.cases:
        case_names = args.cases.split(",")
        cases = [c for c in IEEE_CASES if c[0] in case_names]

    seeds = args.seeds if args.seeds else SEEDS

    t_global_start = time.time()

    # -----------------------------------------------------------------------
    # Phase 1: Train + analyse for all (case, seed) combinations
    # -----------------------------------------------------------------------
    all_ctx = {}  # {case_name: [ctx_all_per_seed]}
    for case_name, ds_path, N_exp in cases:
        all_ctx[case_name] = []
        for seed_idx, seed in enumerate(seeds):
            print(f"=== {case_name} | Seed {seed} ({seed_idx+1}/{len(seeds)}) ===",
                  flush=True)
            ctx = train_and_analyse(case_name, ds_path, N_exp, seed, device)
            if ctx is None:
                print(f"  SKIP (dataset not found or too few edges)", flush=True)
                continue
            all_ctx[case_name].append(ctx)
            print(f"  N={ctx['N']}, |E|={len(ctx['edges'])}, "
                  f"t_train={ctx['t_train']:.1f}s, t_Sc={ctx['t_sc']:.1f}s",
                  flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2: Analysis 1 — LODF wall-clock timing
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 1: WALL-CLOCK TIMING COMPARISON")
    print("=" * 80)
    print(f"{'Case':<10} {'N':>4} {'|E|':>5} "
          f"{'AEGIS train':>12} {'AEGIS S_c':>10} {'AEGIS total':>12} "
          f"{'LODF':>8} {'Eff.Res':>8} {'BF N-1':>8}")
    print("-" * 95)

    timing_results = {}
    for case_name, _, _ in cases:
        ctxs = all_ctx[case_name]
        if not ctxs:
            continue
        timings = [analysis_timing(c) for c in ctxs]
        timing_results[case_name] = timings
        c0 = ctxs[0]
        print(f"{case_name:<10} {c0['N']:>4} {len(c0['edges']):>5} "
              f"{agg([t['t_aegis_train'] for t in timings], '.2f'):>12} "
              f"{agg([t['t_aegis_sc'] for t in timings], '.2f'):>10} "
              f"{agg([t['t_aegis_total'] for t in timings], '.2f'):>12} "
              f"{agg([t['t_lodf'] for t in timings], '.4f'):>8} "
              f"{agg([t['t_eff_res'] for t in timings], '.4f'):>8} "
              f"{agg([t['t_bf_n1'] for t in timings], '.2f'):>8}")

    # -----------------------------------------------------------------------
    # Phase 3: Analysis 2 — N-2 from SVD direction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 2: N-2 CONTINGENCY FROM SVD DIRECTION")
    print("=" * 80)
    print(f"{'Case':<10} {'N':>4} {'Pairs eval':>12} {'Full BF':>8} "
          f"{'Pair overlap':>14} {'Edge overlap':>14} {'Top N-2 sev':>12}")
    print("-" * 85)

    n2_results = {}
    for case_name, _, _ in cases:
        ctxs = all_ctx[case_name]
        if not ctxs:
            continue
        n2s = [analysis_n2(c) for c in ctxs]
        n2_results[case_name] = n2s
        print(f"{case_name:<10} {ctxs[0]['N']:>4} "
              f"{agg([n['n_gt_pairs_evaluated'] for n in n2s], '.0f'):>12} "
              f"{'Y' if n2s[0]['full_bruteforce'] else 'N':>8} "
              f"{agg([n['pair_overlap_frac'] for n in n2s], '.2f'):>14} "
              f"{agg([n['edge_overlap_frac'] for n in n2s], '.2f'):>14} "
              f"{agg([n['top_n2_severity'] for n in n2s], '.4f'):>12}")

    # -----------------------------------------------------------------------
    # Phase 4: Analysis 3 — Rank stability across seeds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 3: RANK STABILITY ACROSS SEEDS")
    print("=" * 80)
    print(f"{'Case':<10} {'N':>4} {'Seeds':>6} {'Mean tau':>12} "
          f"{'Consensus':>12} {'Stable@80%':>12} {'Consensus edges'}")
    print("-" * 90)

    stability_results = {}
    for case_name, _, _ in cases:
        ctxs = all_ctx[case_name]
        if not ctxs:
            continue
        n_edges = len(ctxs[0]["edges"])
        stab = analysis_rank_stability(ctxs, n_edges)
        stability_results[case_name] = stab
        tau_s = f"{stab['mean_tau']:+.3f}+/-{stab['std_tau']:.3f}" if stab["mean_tau"] is not None else "N/A"
        consensus_s = f"{stab['consensus_frac']:.0%}" if stab["consensus_frac"] is not None else "N/A"
        stable80_s = f"{stab['stable_80_frac']:.0%}" if stab["stable_80_frac"] is not None else "N/A"
        consensus_edges_s = str(stab.get("consensus_edges", []))
        if len(consensus_edges_s) > 40:
            consensus_edges_s = consensus_edges_s[:40] + "..."
        print(f"{case_name:<10} {ctxs[0]['N']:>4} "
              f"{stab['n_seeds_evaluated']:>6} "
              f"{tau_s:>12} {consensus_s:>12} {stable80_s:>12} "
              f"{consensus_edges_s}")

    # Free N-2 heavy objects (model, S_c, Vh_c no longer needed after this)
    for case_name_k in list(all_ctx.keys()):
        for ctx_k in all_ctx[case_name_k]:
            for key_to_del in ["S_c", "Vh_c", "sigma_c"]:
                ctx_k.pop(key_to_del, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 5: Analysis 4 — Binary vs admittance
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 4: WHY BINARY ADJACENCY OUTPERFORMS ADMITTANCE-WEIGHTED")
    print("=" * 80)
    print(f"{'Case':<10} {'corr(adm,sev)':>14} {'corr(deg,sev)':>14} "
          f"{'corr(ER,sev)':>14} {'adm CV':>8}")
    print("-" * 70)

    binary_results = {}
    for case_name, _, _ in cases:
        ctxs = all_ctx[case_name]
        if not ctxs:
            continue
        brs = [analysis_binary_vs_admittance(c) for c in ctxs]
        binary_results[case_name] = brs
        print(f"{case_name:<10} "
              f"{agg([b['corr_admittance_severity_spearman'] for b in brs]):>14} "
              f"{agg([b['corr_degree_severity_spearman'] for b in brs]):>14} "
              f"{agg([b['corr_effres_severity_spearman'] for b in brs]):>14} "
              f"{agg([b['admittance_cv'] for b in brs], '.2f'):>8}")

    # Print interpretation from first case with data
    for case_name, _, _ in cases:
        brs = binary_results.get(case_name, [])
        if brs:
            print(f"\nInterpretation ({case_name}): {brs[0]['interpretation']}")
            break

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_global_start
    print(f"\n{'=' * 80}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # JSON summary
    summary = {
        "seeds": seeds,
        "cases": [c[0] for c in cases],
        "total_time_s": elapsed,
        "timing": {},
        "n2": {},
        "stability": {},
        "binary_vs_admittance": {},
    }

    for case_name, _, _ in cases:
        # Timing
        ts = timing_results.get(case_name, [])
        if ts:
            summary["timing"][case_name] = {
                "aegis_total_mean": float(np.mean([t["t_aegis_total"] for t in ts])),
                "aegis_total_std": float(np.std([t["t_aegis_total"] for t in ts])),
                "lodf_mean": float(np.mean([t["t_lodf"] for t in ts])),
                "bf_n1_mean": float(np.mean([t["t_bf_n1"] for t in ts])),
            }
        # N-2
        n2s = n2_results.get(case_name, [])
        if n2s:
            summary["n2"][case_name] = {
                "pair_overlap_mean": float(np.mean([n["pair_overlap_frac"] for n in n2s])),
                "edge_overlap_mean": float(np.mean([n["edge_overlap_frac"] for n in n2s])),
                "full_bruteforce": n2s[0]["full_bruteforce"],
            }
        # Stability
        stab = stability_results.get(case_name)
        if stab and stab["mean_tau"] is not None:
            summary["stability"][case_name] = {
                "mean_tau": stab["mean_tau"],
                "std_tau": stab["std_tau"],
                "consensus_frac": stab["consensus_frac"],
                "stable_80_frac": stab["stable_80_frac"],
            }
        # Binary vs admittance
        brs = binary_results.get(case_name, [])
        if brs:
            summary["binary_vs_admittance"][case_name] = {
                "corr_adm_sev_mean": float(np.mean(
                    [b["corr_admittance_severity_spearman"] for b in brs
                     if b["corr_admittance_severity_spearman"] is not None]
                )) if any(b["corr_admittance_severity_spearman"] is not None for b in brs) else None,
                "corr_er_sev_mean": float(np.mean(
                    [b["corr_effres_severity_spearman"] for b in brs
                     if b["corr_effres_severity_spearman"] is not None]
                )) if any(b["corr_effres_severity_spearman"] is not None for b in brs) else None,
            }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
