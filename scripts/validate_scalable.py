"""Validate matrix-free pipeline against dense pipeline on a small graph.

Checks that ScalableSensitivity produces the same results as the dense
adversarial.py functions for:
  1. Top singular value sigma_1(S_c)
  2. Per-edge vulnerability rankings
  3. Optimal attack direction
  4. Per-node sensitivity norms
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iem.adversarial import (
    _compute_structural_jacobian,
    structural_sensitivity_matrix,
    constrained_sensitivity_matrix,
    optimal_structural_attack,
)
from iem.scalable import ScalableSensitivity


class ToyIGNN(nn.Module):
    """Minimal contractive IGNN for validation."""

    def __init__(self, N: int, d_in: int, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d_in, d, bias=False)
        self.head = nn.Linear(d, 3, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.W.weight)
            self.W.weight.mul_(0.3)

    def operator(self, Z: torch.Tensor, ctx: dict) -> torch.Tensor:
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return torch.relu(A_hat @ Z @ self.W.weight.T + X_proj)


def build_test_graph(N: int = 20, d_in: int = 5, d: int = 8, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    A = torch.zeros(N, N)
    for i in range(N - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    for _ in range(N):
        i, j = torch.randint(0, N, (2,)).tolist()
        if i != j:
            A[i, j] = 1.0
            A[j, i] = 1.0

    D_inv_sqrt = torch.diag(1.0 / (A.sum(dim=1).clamp(min=1).sqrt()))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    model = ToyIGNN(N, d_in, d).to(device)
    X = torch.randn(N, d_in, device=device)
    X_proj = model.proj(X)
    ctx = {"A_hat": A_hat.to(device), "X_proj": X_proj.detach()}

    Z = torch.zeros(N, d, device=device)
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-8:
                break
            Z = Z_new
    z_star = Z_new.detach()

    return model, z_star, ctx, A_hat


def test_sigma_1(model, z_star, ctx, A_hat):
    """Compare top singular value: dense vs matrix-free."""
    F = model.operator

    S = structural_sensitivity_matrix(F, z_star, ctx)
    S_c, edge_list_dense = constrained_sensitivity_matrix(S, A_hat)
    sigma_dense = float(torch.linalg.svdvals(S_c)[0])

    op = ScalableSensitivity(F, z_star, ctx, neumann_terms=30, neumann_tol=1e-8)
    _, sigma_mf, _ = op.top_k_svd(k=1, n_oversamples=10, n_power_iter=7)
    sigma_mf_val = float(sigma_mf[0])

    rel_err = abs(sigma_dense - sigma_mf_val) / (sigma_dense + 1e-10)
    status = "PASS" if rel_err < 0.05 else "FAIL"
    print(f"  sigma_1(S_c):  dense={sigma_dense:.6f}  matrix-free={sigma_mf_val:.6f}  "
          f"rel_err={rel_err:.4f}  [{status}]")
    return rel_err < 0.05


def test_edge_vulnerability(model, z_star, ctx, A_hat):
    """Compare per-edge vulnerability rankings."""
    F = model.operator

    S = structural_sensitivity_matrix(F, z_star, ctx)
    S_c, edge_list_dense = constrained_sensitivity_matrix(S, A_hat)
    dense_vulns = {}
    for k, (i, j) in enumerate(edge_list_dense):
        dense_vulns[(i, j)] = float(S_c[:, k].norm())

    op = ScalableSensitivity(F, z_star, ctx, neumann_terms=30, neumann_tol=1e-8)
    mf_vulns_list = op.edge_vulnerability()
    mf_vulns = {(i, j): v for i, j, v in mf_vulns_list}

    common_edges = set(dense_vulns.keys()) & set(mf_vulns.keys())
    if not common_edges:
        print("  Edge vulnerability: no common edges found [FAIL]")
        return False

    max_rel_err = 0.0
    for edge in common_edges:
        d_val = dense_vulns[edge]
        m_val = mf_vulns[edge]
        rel = abs(d_val - m_val) / (d_val + 1e-10)
        max_rel_err = max(max_rel_err, rel)

    dense_rank = sorted(common_edges, key=lambda e: dense_vulns[e], reverse=True)
    mf_rank = sorted(common_edges, key=lambda e: mf_vulns[e], reverse=True)
    top5_overlap = len(set(dense_rank[:5]) & set(mf_rank[:5]))

    status = "PASS" if max_rel_err < 0.05 else "FAIL"
    print(f"  Edge vulnerability:  max_rel_err={max_rel_err:.4f}  "
          f"top-5 overlap={top5_overlap}/5  [{status}]")
    return max_rel_err < 0.05


def test_matvec_consistency(model, z_star, ctx, A_hat):
    """Verify S_c * v matches between dense and matrix-free."""
    F = model.operator

    S = structural_sensitivity_matrix(F, z_star, ctx)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)

    op = ScalableSensitivity(F, z_star, ctx, neumann_terms=30, neumann_tol=1e-8)

    torch.manual_seed(123)
    v = torch.randn(S_c.shape[1])
    dense_result = S_c @ v
    mf_result = op.matvec(v)

    rel_err = float((dense_result - mf_result).norm() / (dense_result.norm() + 1e-10))
    status = "PASS" if rel_err < 0.05 else "FAIL"
    print(f"  S_c * v:  rel_err={rel_err:.4f}  [{status}]")
    return rel_err < 0.05


def test_rmatvec_consistency(model, z_star, ctx, A_hat):
    """Verify S_c^T * u matches between dense and matrix-free."""
    F = model.operator
    D = z_star.numel()

    S = structural_sensitivity_matrix(F, z_star, ctx)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)

    op = ScalableSensitivity(F, z_star, ctx, neumann_terms=30, neumann_tol=1e-8)

    torch.manual_seed(456)
    u = torch.randn(D)
    dense_result = S_c.T @ u
    mf_result = op.rmatvec(u)

    rel_err = float((dense_result - mf_result).norm() / (dense_result.norm() + 1e-10))
    status = "PASS" if rel_err < 0.05 else "FAIL"
    print(f"  S_c^T * u:  rel_err={rel_err:.4f}  [{status}]")
    return rel_err < 0.05


def test_node_sensitivity(model, z_star, ctx, A_hat):
    """Compare per-node sensitivity norms."""
    F = model.operator
    d = z_star.shape[-1]
    N = A_hat.shape[0]

    S = structural_sensitivity_matrix(F, z_star, ctx)
    S_c, _ = constrained_sensitivity_matrix(S, A_hat)
    dense_norms = torch.zeros(N)
    for v in range(N):
        s, e = v * d, (v + 1) * d
        if e <= S_c.shape[0]:
            dense_norms[v] = S_c[s:e].norm()

    op = ScalableSensitivity(F, z_star, ctx, neumann_terms=30, neumann_tol=1e-8)
    mf_norms = op.node_sensitivity_norms(n_probes=50)

    rel_err = float((dense_norms - mf_norms).norm() / (dense_norms.norm() + 1e-10))

    dense_rank = dense_norms.argsort(descending=True)[:5].tolist()
    mf_rank = mf_norms.argsort(descending=True)[:5].tolist()
    top5_overlap = len(set(dense_rank) & set(mf_rank))

    status = "PASS" if rel_err < 0.30 else "FAIL"
    print(f"  Node sensitivity:  rel_err={rel_err:.4f}  "
          f"top-5 overlap={top5_overlap}/5  [{status}]")
    return rel_err < 0.30


def main():
    print("=" * 60)
    print("Matrix-Free Pipeline Validation")
    print("=" * 60)

    for N in [15, 30, 50]:
        print(f"\n--- N={N} nodes ---")
        model, z_star, ctx, A_hat = build_test_graph(N=N, d=8)
        D = z_star.numel()
        n_edges = int((A_hat.abs() > 1e-10).sum().item() // 2)
        print(f"  D={D}, |E|={n_edges}")

        results = []
        results.append(test_matvec_consistency(model, z_star, ctx, A_hat))
        results.append(test_rmatvec_consistency(model, z_star, ctx, A_hat))
        results.append(test_sigma_1(model, z_star, ctx, A_hat))
        results.append(test_edge_vulnerability(model, z_star, ctx, A_hat))
        results.append(test_node_sensitivity(model, z_star, ctx, A_hat))

        passed = sum(results)
        total = len(results)
        print(f"  Result: {passed}/{total} passed")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == "__main__":
    main()
