"""Revision-R2 P3.10 — apply S_c framework to robust-GNN backbones.

Implements minimal RobustGCN (Zhu 2019, Gaussian variational layer) and
GNNGuard-lite (Zhang 2020, similarity-thresholded attention prune) backbones,
applies the AEGIS S_c pipeline to each, and reports per-edge ranking tau vs
brute-force discrete ground truth on Cora and Citeseer.

Hypothesis: robust backbones produce flatter S_c rankings (smaller singular
gap, lower kappa) because they explicitly dampen sensitivity. If true, this
supports the position that S_c diagnoses what the architecture itself
attempts to defend against.

Closes: P3.10 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_10_robust_arch.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_R2._common import (
    SEEDS,
    forward_and_subgraph,
    full_graph_ctx_Z,
    load_dataset,
    reconverge,
    train_ignn,
)

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)

SUBGRAPH_N = 50
OUT_CSV = Path("results/revision_R2/robust_arch.csv")


SPECTRAL_KAPPA_TARGET = 0.9  # subcritical safety margin


@torch.no_grad()
def spectral_normalize_(linear: nn.Linear, target: float = SPECTRAL_KAPPA_TARGET):
    """In-place cap sigma_1(linear.weight) <= target via direct SVD rescale.

    Used instead of nn.utils.spectral_norm because both backbones access
    ``W.weight.t()`` directly in their forward (not ``W(x)``), so the
    forward_pre_hook that triggers sigma_norm power iteration never fires
    for W. Called at init AND post-train to guarantee kappa <= target at
    AEGIS-analysis time.
    """
    sigma = torch.linalg.svdvals(linear.weight)[0].item()
    if sigma > target:
        linear.weight.data.mul_(target / max(sigma, 1e-12))


class RobustGCNLite(nn.Module):
    """Minimal RobustGCN: Gaussian-variational mean+var layers + hidden recurrence.

    Architecture has TWO linear stages — ``W_mu`` is the n_in -> hidden input
    projection (its weight is shape [hidden, n_in] and cannot be used in the
    hidden-space recurrence ``A @ z @ W``) and ``W_hidden`` is the
    hidden -> hidden recurrent map. Both ``operator`` and ``forward`` use
    ``W_hidden`` so the S_c Jacobian computed at the forward's Z value is
    self-consistent with the architecture.

    ``W_hidden`` is spectral-capped to sigma_1 <= ``SPECTRAL_KAPPA_TARGET``
    (0.9) at init and again post-train (mirrors IGNN's spectral-norm
    constraint via Miyato et al. 2018). Without this constraint
    round-2 results showed kappa drifting > 1 (RobustGCN ~= 1.0,
    GNNGuard ~= 2.0), pushing the operator into the supercritical regime
    where Theorem 1's first-order bound no longer applies; that is why
    the unconstrained run produced anti-correlated AEGIS rankings.
    """
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.W_mu = nn.Linear(n_in, hidden, bias=False)
        self.W_sigma = nn.Linear(n_in, hidden, bias=False)
        self.W_hidden = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_out)
        spectral_normalize_(self.W_hidden)

    def operator(self, z, ctx):  # for S_c compatibility
        A = ctx["A_hat"]
        return F.relu(A @ z @ self.W_hidden.weight.t() + ctx.get("X_proj", 0))

    def forward(self, X, A_hat):
        X_proj = self.W_mu(X)
        mu = F.relu(A_hat @ X_proj)
        sigma = F.softplus(A_hat @ self.W_sigma(X)) + 1e-6  # variational scale (unused for forward output)
        Z = F.relu(A_hat @ mu @ self.W_hidden.weight.t() + X_proj)
        logits = self.head(Z)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}
        return logits, Z, ctx


class GNNGuardLite(nn.Module):
    """GNNGuard-lite: prune low-similarity edges via cosine threshold.

    ``W2`` is spectral-capped at init AND post-train (see
    ``spectral_normalize_``); the forward accesses ``W2.weight.t()``
    directly rather than calling ``W2(x)``, so ``nn.utils.spectral_norm``'s
    forward-hook would not fire here. Manual SVD rescale guarantees
    sigma_1(W2) <= ``SPECTRAL_KAPPA_TARGET`` at AEGIS-analysis time.
    """
    def __init__(self, n_in, hidden, n_out, prune_thresh=0.1):
        super().__init__()
        self.W1 = nn.Linear(n_in, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_out)
        self.prune_thresh = prune_thresh
        spectral_normalize_(self.W2)

    def operator(self, z, ctx):
        A = ctx["A_hat"]
        return F.relu(A @ z @ self.W2.weight.t() + ctx.get("X_proj", 0))

    def forward(self, X, A_hat):
        X_proj = self.W1(X)
        H = F.relu(A_hat @ X_proj)
        H_norm = F.normalize(H, dim=1)
        sim = H_norm @ H_norm.t()
        prune_mask = (sim > self.prune_thresh).float()
        A_pruned = A_hat * prune_mask
        Z = F.relu(A_pruned @ H @ self.W2.weight.t())
        ctx = {"A_hat": A_pruned, "X_proj": X_proj}
        return self.head(Z), Z, ctx


def train(model, X, A_hat, y, train_mask, n_epochs=200, lr=0.01):
    """Train with per-step spectral-cap projection on the hidden recurrence.

    The S_c framework requires kappa <= 1 at AEGIS-analysis time, so we
    re-apply ``spectral_normalize_`` to the hidden-recurrence weight after
    every gradient step (cheap O(h^3) SVD on a 64x64 matrix).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    if isinstance(model, RobustGCNLite):
        hidden_W = model.W_hidden
    elif isinstance(model, GNNGuardLite):
        hidden_W = model.W2
    else:
        hidden_W = None
    for _ in range(n_epochs):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if hidden_W is not None:
            spectral_normalize_(hidden_W)
    model.eval()
    # Final post-train cap (paranoia: catches any FP drift).
    if hidden_W is not None:
        spectral_normalize_(hidden_W)
    return model


def aegis_ranking(model, X_sub, A_sub):
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(F_op, Z, ctx)
    S = structural_sensitivity_matrix(F_op, Z, ctx, J_z=J_z, J_A=J_A)
    S_c, edges = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return [], np.array([]), float("nan"), float("nan")
    v = S_c.norm(dim=0).cpu().numpy()
    order = np.argsort(-v)
    sigma = torch.linalg.svdvals(S_c).cpu().numpy()
    gap = (sigma[0] - sigma[1]) / sigma[0] if len(sigma) > 1 else float("nan")
    kappa = float(torch.linalg.svdvals(J_z)[0].item())
    return [edges[i] for i in order], v[order], gap, kappa


def damage(model, X, A0, A_p):
    with torch.no_grad():
        _, Z0, _ = model(X, A0)
        _, Zp, _ = model(X, A_p)
    return float((Zp - Z0).norm().item())


def discrete_ground_truth(model, X_sub, A_sub):
    """Per-edge brute-force single-removal damage."""
    N = A_sub.shape[0]
    out = {}
    for i in range(N):
        for j in range(i + 1, N):
            if float(A_sub[i, j].item()) <= 0:
                continue
            A_p = A_sub.clone()
            A_p[i, j] = 0.0; A_p[j, i] = 0.0
            out[(i, j)] = damage(model, X_sub, A_sub, A_p)
    edges = list(out.keys())
    scores = np.array([out[e] for e in edges])
    return edges, scores


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dname in ["Cora", "Citeseer"]:
        for seed in SEEDS:
            X, A_hat, y, train_mask, n_features, n_classes = load_dataset(dname)
            X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
            train_mask = train_mask.to(device)
            for arch_name, arch_cls in [
                ("RobustGCN_lite", RobustGCNLite),
                ("GNNGuard_lite",  GNNGuardLite),
            ]:
                torch.manual_seed(seed)
                model = arch_cls(X.shape[1], 64, n_classes).to(device)
                model = train(model, X, A_hat, y, train_mask)
                X_sub, A_sub, Z_sub, ctx_sub, _ctx_full, _Z_full, idx = forward_and_subgraph(model, X, A_hat, max_nodes=SUBGRAPH_N)
                aegis_e, v_aegis, gap, kappa = aegis_ranking(model, X_sub, A_sub)
                gt_edges, gt_scores = discrete_ground_truth(model, X_sub, A_sub)
                # Build edge -> rank dicts (rank 0 = most vulnerable / most damaging).
                # NOTE: the previous version compared `aegis_rank[e]` against
                # `gt_edges.index(e)` -- the row-major iteration order of edges,
                # not the damage-sorted rank. This made every reported tau
                # uninterpretable. Fixed to mirror R2_01_grbcd_baseline's pattern.
                aegis_edge_to_rank = {e: r for r, e in enumerate(aegis_e)}
                gt_edges_sorted = [gt_edges[i] for i in np.argsort(-gt_scores)]
                gt_edge_to_rank = {e: r for r, e in enumerate(gt_edges_sorted)}
                common = list(set(aegis_edge_to_rank) & set(gt_edge_to_rank))
                if len(common) < 3:
                    tau, p_tau = float("nan"), float("nan")
                else:
                    a = np.array([aegis_edge_to_rank[e] for e in common])
                    g = np.array([gt_edge_to_rank[e] for e in common])
                    tau, p_tau = kendalltau(a, g)
                with torch.no_grad():
                    logits, _, _ = model(X, A_hat)
                    acc = float((logits.argmax(1) == y).float().mean().item())
                rows.append({
                    "dataset": dname,
                    "architecture": arch_name,
                    "seed": seed,
                    "test_accuracy": acc,
                    "kappa_Jz": kappa,
                    "singular_gap_sigma1_minus_sigma2_over_sigma1": gap,
                    "tau_aegis_vs_brute": float(tau),
                    "tau_pvalue": float(p_tau),
                })
                print(f"  {dname:8s} {arch_name:15s} seed={seed:5d} "
                      f"acc={acc:.3f} kappa={kappa:.3f} "
                      f"gap={gap:.3f} tau={tau:+.3f}", flush=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
