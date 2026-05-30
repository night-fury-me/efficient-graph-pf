"""Diagnose the 1.2% relative error in sigma_1(S_c) between dense and
matrix-free AEGIS pipelines at N=200 (IGNN, rho~0.96).

Isolates error contributions from:
  A. Neumann series truncation (dominant suspect at rho=0.96)
  B. Structural JVP: autograd vs finite-difference
  C. Randomized SVD accuracy
  D. Edge list consistency

Run: .venv/bin/python scripts/diagnose_error.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iem.adversarial import (
    _compute_structural_jacobian,
    structural_sensitivity_matrix,
    constrained_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity
from iem.certify import spectral_radius


# ---------------------------------------------------------------
# Model & graph setup (reuse the ToyIGNN from validate_scalable)
# ---------------------------------------------------------------

class ToyIGNN(nn.Module):
    """Contractive IGNN with tunable spectral radius."""

    def __init__(self, N: int, d_in: int, d: int, rho_target: float = 0.96):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d_in, d, bias=False)
        self.head = nn.Linear(d, 3, bias=False)
        self.rho_target = rho_target
        with torch.no_grad():
            nn.init.orthogonal_(self.W.weight)
            # Scale W so that rho(sigma'(.) * (A_hat x W)) ~ rho_target.
            # We'll calibrate after building the graph.

    def operator(self, Z: torch.Tensor, ctx: dict) -> torch.Tensor:
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return torch.relu(A_hat @ Z @ self.W.weight.T + X_proj)


def _converge(model, ctx, N, d, device, max_iter=1000):
    """Converge to fixed point from zero init."""
    Z = torch.zeros(N, d, device=device)
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-9:
                break
            Z = Z_new
    return Z_new.detach()


def _measure_rho(model, z_star, ctx):
    """Measure spectral radius at current fixed point."""
    return spectral_radius(
        lambda z: model.operator(z.reshape(z_star.shape), ctx).reshape(-1),
        z_star,
    )


def build_graph(N: int, d_in: int = 5, d: int = 8, rho_target: float = 0.96,
                seed: int = 42):
    """Build a connected graph with N nodes and tune W for target rho.

    Uses binary search on ||W|| scale because the mapping scale -> rho
    is nonlinear (ReLU changes the Jacobian sparsity at each fixed point).
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Erdos-Renyi-ish graph with guaranteed connectivity
    A = torch.zeros(N, N)
    for i in range(N - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    n_extra = int(N * 1.5)
    for _ in range(n_extra):
        i, j = torch.randint(0, N, (2,)).tolist()
        if i != j:
            A[i, j] = 1.0
            A[j, i] = 1.0

    deg = A.sum(dim=1).clamp(min=1)
    D_inv_sqrt = torch.diag(1.0 / deg.sqrt())
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    model = ToyIGNN(N, d_in, d, rho_target).to(device)
    X = torch.randn(N, d_in, device=device)
    X_proj = model.proj(X).detach()
    ctx = {"A_hat": A_hat.to(device), "X_proj": X_proj}

    # Save base W direction (orthogonal init)
    W_base = model.W.weight.data.clone()

    # Binary search for scale that gives rho_target
    lo, hi = 0.01, 5.0
    best_scale, best_rho, best_z = 0.1, 0.0, None

    for iteration in range(40):
        mid = (lo + hi) / 2.0
        with torch.no_grad():
            model.W.weight.copy_(W_base * mid)
        z_star = _converge(model, ctx, N, d, device)
        rho = _measure_rho(model, z_star, ctx)

        if best_z is None or abs(rho - rho_target) < abs(best_rho - rho_target):
            best_scale, best_rho, best_z = mid, rho, z_star.clone()

        if rho < rho_target:
            lo = mid
        else:
            hi = mid

        if abs(rho - rho_target) < 0.003:
            break

    # Apply best scale
    with torch.no_grad():
        model.W.weight.copy_(W_base * best_scale)
    z_star = _converge(model, ctx, N, d, device)
    rho_actual = _measure_rho(model, z_star, ctx)

    n_edges = int((A_hat.abs() > 1e-10).float().triu(diagonal=1).sum().item())
    D_total = z_star.numel()
    print(f"Graph: N={N}, d={d}, D={D_total}, |E|={n_edges}")
    print(f"Target rho={rho_target}, actual rho={rho_actual:.4f} (W_scale={best_scale:.4f})")

    return model, z_star, ctx, A_hat, rho_actual


# ---------------------------------------------------------------
# Test A: Neumann convergence analysis
# ---------------------------------------------------------------

def test_neumann_convergence(model, z_star, ctx, rho):
    """Print ||J_z^k * b|| for k=0..K_max to see convergence behavior."""
    print("\n" + "=" * 65)
    print("TEST A: Neumann series convergence")
    print("=" * 65)

    F = model.operator
    N = ctx["A_hat"].shape[0]
    D = z_star.numel()

    # Build a representative RHS: structural JVP for edge 0
    A_hat = ctx["A_hat"]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_hat[i, j].abs() > 1e-10:
                edges.append((i, j))
    i0, j0 = edges[0]
    delta_A = torch.zeros(N, N)
    delta_A[i0, j0] = 1.0
    delta_A[j0, i0] = 1.0

    # Get the structural JVP as the RHS
    from torch.func import jvp as torch_jvp
    A_base = A_hat.detach()

    def F_of_A(A_val):
        ctx_local = {**ctx, "A_hat": A_val}
        return F(z_star, ctx_local).reshape(-1)

    _, rhs = torch_jvp(F_of_A, (A_base,), (delta_A,))

    # Now iterate J_z^k * rhs
    z_flat = z_star.reshape(-1).detach()

    def jvp_Jz(v):
        def F_flat(z):
            return F(z.reshape(z_star.shape), ctx).reshape(-1)
        _, Jv = torch_jvp(F_flat, (z_flat,), (v,))
        return Jv

    term = rhs.clone()
    b_norm = rhs.norm().item()
    partial_sum = rhs.clone()

    # Dense ground truth: (I - J_z)^{-1} * rhs
    from iem.ift import compute_jacobian
    def F_z(z):
        return F(z.reshape(z_star.shape), ctx).reshape(-1)
    J_z = compute_jacobian(F_z, z_star)
    I = torch.eye(D)
    exact_solution = torch.linalg.solve(I - J_z, rhs)
    exact_norm = exact_solution.norm().item()

    print(f"\nrho(J_z) = {rho:.4f}")
    print(f"||b|| = {b_norm:.6f}")
    print(f"||(I-J_z)^{{-1}} b||_exact = {exact_norm:.6f}")
    print(f"Predicted convergence: rho^k ~ {rho:.2f}^k")
    print(f"After K=20: rho^20 = {rho**20:.4f}")
    print(f"After K=50: rho^50 = {rho**50:.6f}")
    print(f"After K=100: rho^100 = {rho**100:.8f}")
    print()
    print(f"{'k':>4}  {'||J^k b||':>14}  {'||J^k b||/||b||':>16}  "
          f"{'rel_err_partial':>16}  {'converged_tol':>14}")
    print("-" * 75)

    K_max = 80
    for k in range(K_max):
        if k > 0:
            term = jvp_Jz(term)
            partial_sum = partial_sum + term

        term_norm = term.norm().item()
        rel_term = term_norm / b_norm
        rel_err = (partial_sum - exact_solution).norm().item() / exact_norm
        converged = "YES" if term_norm < 1e-6 * b_norm else ""

        if k <= 25 or k % 5 == 0 or converged:
            print(f"{k:4d}  {term_norm:14.6e}  {rel_term:16.6e}  "
                  f"{rel_err:16.6e}  {converged:>14}")

        if term_norm < 1e-10 * b_norm:
            print(f"  ... fully converged at k={k}")
            break

    # Report: what K is needed for 0.1% accuracy?
    term2 = rhs.clone()
    partial2 = rhs.clone()
    for k in range(1, 500):
        term2 = jvp_Jz(term2)
        partial2 = partial2 + term2
        rel = (partial2 - exact_solution).norm().item() / exact_norm
        if rel < 0.001:
            print(f"\n  K needed for 0.1% accuracy: {k}")
            break
    else:
        print(f"\n  K=500 not sufficient for 0.1% accuracy")

    # What's the Neumann result at K=20?
    term3 = rhs.clone()
    partial3 = rhs.clone()
    for k in range(1, 21):
        term3 = jvp_Jz(term3)
        partial3 = partial3 + term3
    rel_at_20 = (partial3 - exact_solution).norm().item() / exact_norm
    print(f"  Neumann (K=20) rel_err vs exact solve: {rel_at_20:.6f} ({rel_at_20*100:.2f}%)")

    return J_z, exact_solution, rhs


# ---------------------------------------------------------------
# Test B: Structural JVP — autograd vs finite-difference
# ---------------------------------------------------------------

def test_structural_jvp(model, z_star, ctx):
    """Compare autograd JVP vs finite-difference for a single edge."""
    print("\n" + "=" * 65)
    print("TEST B: Structural JVP accuracy (autograd vs finite-diff)")
    print("=" * 65)

    F = model.operator
    A_hat = ctx["A_hat"]
    N = A_hat.shape[0]

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_hat[i, j].abs() > 1e-10:
                edges.append((i, j))

    # Test on first 5 edges
    for idx, (i, j) in enumerate(edges[:5]):
        delta_A = torch.zeros(N, N)
        delta_A[i, j] = 1.0
        delta_A[j, i] = 1.0

        # Method 1: autograd JVP (what scalable.py uses)
        from torch.func import jvp as torch_jvp
        A_base = A_hat.detach()

        def F_of_A(A_val):
            ctx_local = {**ctx, "A_hat": A_val}
            return F(z_star, ctx_local).reshape(-1)

        _, jvp_autograd = torch_jvp(F_of_A, (A_base,), (delta_A,))

        # Method 2: finite-difference (what adversarial.py uses)
        eps = 1e-4
        with torch.no_grad():
            f_base = F(z_star, ctx).reshape(-1)
            A_pert = A_hat.clone()
            A_pert[i, j] += eps
            # NOTE: dense pipeline perturbs ONE entry, not symmetric!
            f_pert = F(z_star, {**ctx, "A_hat": A_pert}).reshape(-1)
            fd_ij = (f_pert - f_base) / eps

            # For the symmetric column, dense does S[:,i*N+j] + S[:,j*N+i]
            A_pert2 = A_hat.clone()
            A_pert2[j, i] += eps
            f_pert2 = F(z_star, {**ctx, "A_hat": A_pert2}).reshape(-1)
            fd_ji = (f_pert2 - f_base) / eps

            fd_symmetric = fd_ij + fd_ji

        rel_err = (jvp_autograd - fd_symmetric).norm().item() / (fd_symmetric.norm().item() + 1e-15)
        print(f"  Edge ({i},{j}):  ||autograd - fd_sym|| / ||fd_sym|| = {rel_err:.6e}  "
              f"||autograd||={jvp_autograd.norm():.4f}  ||fd_sym||={fd_symmetric.norm():.4f}")


# ---------------------------------------------------------------
# Test C: Randomized SVD accuracy
# ---------------------------------------------------------------

def test_randomized_svd(model, z_star, ctx, A_hat, J_z):
    """Compare randomized SVD with exact SVD of S_c."""
    print("\n" + "=" * 65)
    print("TEST C: Randomized SVD accuracy")
    print("=" * 65)

    F = model.operator
    N = A_hat.shape[0]
    D = z_star.numel()

    # Dense ground truth
    S = structural_sensitivity_matrix(F, z_star, ctx, J_z=J_z)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)
    sigma_exact = torch.linalg.svdvals(S_c)
    sigma_1_exact = float(sigma_exact[0])
    print(f"\n  Dense sigma_1(S_c) = {sigma_1_exact:.6f}")
    print(f"  S_c shape: {S_c.shape}")

    # Matrix-free with EXACT Neumann (K=200, should converge)
    # But first test with high K to isolate rSVD error from Neumann error
    for K_neumann in [20, 50, 100, 200]:
        op = ScalableSensitivity(F, z_star, ctx, neumann_terms=K_neumann, neumann_tol=1e-10)

        # First test: single column comparison to isolate Neumann error
        if len(op.edge_list) > 0:
            col_mf = op._column(0)
            col_dense = S_c[:, 0]
            col_rel = (col_mf - col_dense).norm().item() / (col_dense.norm().item() + 1e-15)
        else:
            col_rel = float('nan')

        # Now rSVD
        torch.manual_seed(0)
        _, sigma_mf, _ = op.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)
        sigma_1_mf = float(sigma_mf[0])
        rsvd_rel = abs(sigma_1_mf - sigma_1_exact) / sigma_1_exact

        print(f"  K={K_neumann:3d}:  col_0_rel_err={col_rel:.6e}  "
              f"sigma_1_mf={sigma_1_mf:.6f}  rsvd_rel_err={rsvd_rel:.6e}")

    # Also test rSVD with varying power iterations at fixed K=200
    print(f"\n  Varying n_power_iter (K=200 Neumann):")
    op200 = ScalableSensitivity(F, z_star, ctx, neumann_terms=200, neumann_tol=1e-10)
    for npi in [0, 1, 2, 3, 5, 7, 10]:
        torch.manual_seed(0)
        _, sig, _ = op200.top_k_svd(k=6, n_oversamples=10, n_power_iter=npi)
        rel = abs(float(sig[0]) - sigma_1_exact) / sigma_1_exact
        print(f"    n_power_iter={npi:2d}:  sigma_1={float(sig[0]):.6f}  rel_err={rel:.6e}")

    return sigma_1_exact


# ---------------------------------------------------------------
# Test D: Edge list consistency
# ---------------------------------------------------------------

def test_edge_list(model, z_star, ctx, A_hat):
    """Verify dense and matrix-free enumerate the same edges."""
    print("\n" + "=" * 65)
    print("TEST D: Edge list consistency")
    print("=" * 65)

    N = A_hat.shape[0]

    # Dense edge list (from constrained_sensitivity_matrix)
    dense_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_hat[i, j].abs() > 1e-10:
                dense_edges.append((i, j))

    # Matrix-free edge list
    op = ScalableSensitivity(model.operator, z_star, ctx)
    mf_edges = op.edge_list

    match = set(dense_edges) == set(mf_edges)
    print(f"  Dense edges: {len(dense_edges)}")
    print(f"  Matrix-free edges: {len(mf_edges)}")
    print(f"  Sets match: {match}")

    if not match:
        only_dense = set(dense_edges) - set(mf_edges)
        only_mf = set(mf_edges) - set(dense_edges)
        if only_dense:
            print(f"  Only in dense: {list(only_dense)[:10]}")
        if only_mf:
            print(f"  Only in matrix-free: {list(only_mf)[:10]}")


# ---------------------------------------------------------------
# Test E: Full column comparison (isolates Neumann + JVP error)
# ---------------------------------------------------------------

def test_column_comparison(model, z_star, ctx, A_hat, J_z):
    """Compare individual columns of S_c: dense vs matrix-free."""
    print("\n" + "=" * 65)
    print("TEST E: Column-by-column S_c comparison")
    print("=" * 65)

    F = model.operator
    S = structural_sensitivity_matrix(F, z_star, ctx, J_z=J_z)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)

    n_test = min(10, S_c.shape[1])

    for K in [20, 50, 100]:
        op = ScalableSensitivity(F, z_star, ctx, neumann_terms=K, neumann_tol=1e-10)
        errs = []
        for idx in range(n_test):
            col_mf = op._column(idx)
            col_dense = S_c[:, idx]
            rel = (col_mf - col_dense).norm().item() / (col_dense.norm().item() + 1e-15)
            errs.append(rel)
        mean_err = sum(errs) / len(errs)
        max_err = max(errs)
        print(f"  Neumann K={K:3d}:  mean_col_rel_err={mean_err:.6e}  "
              f"max_col_rel_err={max_err:.6e}")


# ---------------------------------------------------------------
# Summary: error budget breakdown
# ---------------------------------------------------------------

def error_budget(model, z_star, ctx, A_hat, J_z, sigma_1_exact, rho):
    """Quantify the error contribution from each source."""
    print("\n" + "=" * 65)
    print("ERROR BUDGET BREAKDOWN")
    print("=" * 65)

    F = model.operator

    # 1. Neumann error at K=20 (the default)
    op20 = ScalableSensitivity(F, z_star, ctx, neumann_terms=20, neumann_tol=1e-10)
    torch.manual_seed(0)
    _, sig20, _ = op20.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)
    err_neumann_20 = abs(float(sig20[0]) - sigma_1_exact) / sigma_1_exact

    # 2. Neumann error at K=200 (should converge fully)
    op200 = ScalableSensitivity(F, z_star, ctx, neumann_terms=200, neumann_tol=1e-10)
    torch.manual_seed(0)
    _, sig200, _ = op200.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)
    err_converged = abs(float(sig200[0]) - sigma_1_exact) / sigma_1_exact

    # 3. rSVD error = (error with converged Neumann) since Neumann is exact
    err_rsvd = err_converged  # remaining error after Neumann converges

    # 4. Structural JVP error (autograd vs FD) — already shown to be ~O(eps)

    print(f"\n  rho = {rho:.4f}")
    print(f"  sigma_1(S_c) exact = {sigma_1_exact:.6f}")
    print(f"")
    print(f"  Total error (K=20, default pipeline): {err_neumann_20:.6e} ({err_neumann_20*100:.3f}%)")
    print(f"  Neumann error (K=20 vs converged):    {abs(err_neumann_20 - err_rsvd):.6e} ({abs(err_neumann_20 - err_rsvd)*100:.3f}%)")
    print(f"  rSVD error (converged Neumann vs exact): {err_rsvd:.6e} ({err_rsvd*100:.3f}%)")
    print(f"  Structural JVP error: autograd is exact (0%), FD has O(1e-4) truncation")
    print(f"  Note: autograd JVP in matrix-free is MORE accurate than FD in dense!")
    print()

    # Recommendation
    if err_neumann_20 > 0.005:
        # Compute K needed
        S = structural_sensitivity_matrix(F, z_star, ctx, J_z=J_z)
        S_c, _ = constrained_sensitivity_matrix(S, A_hat)

        best_K = 20
        for K_try in [30, 40, 50, 60, 80, 100, 150, 200]:
            op_try = ScalableSensitivity(F, z_star, ctx, neumann_terms=K_try, neumann_tol=1e-10)
            torch.manual_seed(0)
            _, sig_try, _ = op_try.top_k_svd(k=6, n_oversamples=10, n_power_iter=5)
            err_try = abs(float(sig_try[0]) - sigma_1_exact) / sigma_1_exact
            if err_try < 0.001:
                best_K = K_try
                break

        print(f"  RECOMMENDATION: Increase neumann_terms from 20 to {best_K}")
        print(f"  Formula: K >= log(tol) / log(rho) = log(1e-6) / log({rho:.2f}) = "
              f"{-6 * 2.302585 / (torch.tensor(rho).log().item()):.0f}")
        K_formula = int(-6 * 2.302585 / torch.tensor(rho).log().item()) + 1
        print(f"  Suggested adaptive K: ceil(-log(tol)/log(rho)) = {K_formula}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 65)
    print("DIAGNOSING 1.2% ERROR IN sigma_1(S_c): DENSE vs MATRIX-FREE")
    print("=" * 65)

    # Build graph at N=200 with rho~0.96
    N = 200
    model, z_star, ctx, A_hat, rho = build_graph(N=N, d=8, rho_target=0.96)

    # Test A: Neumann convergence
    J_z, exact_col, rhs = test_neumann_convergence(model, z_star, ctx, rho)

    # Test B: Structural JVP
    test_structural_jvp(model, z_star, ctx)

    # Test D: Edge list
    test_edge_list(model, z_star, ctx, A_hat)

    # Test E: Column comparison (uses J_z from test A)
    test_column_comparison(model, z_star, ctx, A_hat, J_z)

    # Test C: Randomized SVD
    sigma_1_exact = test_randomized_svd(model, z_star, ctx, A_hat, J_z)

    # Error budget
    error_budget(model, z_star, ctx, A_hat, J_z, sigma_1_exact, rho)

    print("\n" + "=" * 65)
    print("DIAGNOSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
