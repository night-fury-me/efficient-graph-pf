"""Debug X4: is the low fixed-vs-recomputed tau a subgraph-restriction artifact?

Compares two setups on Cora seed 42:
  (A) RESTRICTED base A_sub = A_hat[idx][:,idx] (uses full-graph degrees, only subgraph
      neighbors present). Recomputed deletion uses full-graph degree (current X4).
  (B) SELF-CONSISTENT base A0 = normalize(reconstructed raw subgraph) (subgraph degrees).
      Recomputed deletion uses subgraph degree. Both deletions consistent with A0.

Also reports tau vs degree to test the O(1/d) claim.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iem.adversarial import extract_ego_subgraph
from iem.examples.ignn_cora import _load_cora
from exp_phase_transition import set_seed, train_ignn_kappa


def reconverge(model, A, X_proj, z0, iters=400, tol=1e-9):
    ctx = {"A_hat": A, "X_proj": X_proj}
    Z = z0.clone()
    with torch.no_grad():
        for _ in range(iters):
            Zn = model.operator(Z, ctx)
            if (Zn - Z).norm() < tol:
                break
            Z = Zn
    return Zn


def normalize(A_raw):
    """D^-1/2 (A_raw + I) D^-1/2 with subgraph degrees."""
    N = A_raw.shape[0]
    AI = A_raw + torch.eye(N, device=A_raw.device, dtype=A_raw.dtype)
    d = AI.sum(1)
    c = d.clamp(min=1e-12).rsqrt()
    return c[:, None] * AI * c[None, :]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    data = _load_cora(Path("datasets/cora"))
    model, Z_star, ctx, _ = train_ignn_kappa(data, device, 42, 0.90)
    A_hat = data["A_hat"].to(device)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx].clone()
    X_proj_sub = ctx["X_proj"][idx].clone()
    N = A_sub.shape[0]

    # reconstruct raw subgraph adjacency (binary, no self-loop)
    c_full = (1.0 / torch.diagonal(A_sub).clamp(min=1e-12)).sqrt()   # sqrt(d_full+1)
    AI_full = A_sub * c_full[:, None] * c_full[None, :]
    A_raw = (AI_full > 0.5).to(A_sub.dtype)
    A_raw.fill_diagonal_(0)
    deg_sub = A_raw.sum(1)                                            # subgraph degree
    deg_full = c_full ** 2 - 1.0                                      # full-graph degree

    A0 = normalize(A_raw)                                             # self-consistent base
    c_sub = (1.0 / torch.diagonal(A0).clamp(min=1e-12)).sqrt()       # sqrt(d_sub+1)

    Z_sub = reconverge(model, A_sub, X_proj_sub, Z_star[idx].clone())  # eq on restricted
    Z0 = reconverge(model, A0, X_proj_sub, Z_star[idx].clone())        # eq on self-consistent

    ii, jj = torch.where(torch.triu((A_raw > 0.5), diagonal=1))
    edges = list(zip(ii.tolist(), jj.tolist()))
    print(f"edges={len(edges)}  deg_sub: min={int(deg_sub.min())} max={int(deg_sub.max())} "
          f"mean={float(deg_sub.mean()):.1f}   deg_full: min={int(deg_full.min())} "
          f"max={int(deg_full.max())} mean={float(deg_full.mean()):.1f}")

    def rescale_delete(A, c, i, j):
        ci2, cj2 = float(c[i])**2, float(c[j])**2
        if ci2 - 1 <= 1e-9 or cj2 - 1 <= 1e-9:
            return None
        rho = torch.ones(N, device=A.device, dtype=A.dtype)
        rho[i] = (ci2 / (ci2 - 1))**0.5
        rho[j] = (cj2 / (cj2 - 1))**0.5
        Ar = A * rho[:, None] * rho[None, :]
        Ar[i, j] = 0.0; Ar[j, i] = 0.0
        return Ar

    A_fixed_A, A_recomp_A, A_fixed_B, A_recomp_B = [], [], [], []
    min_deg_full, min_deg_sub = [], []
    for (i, j) in edges:
        # (A) restricted base, full-graph-degree recomputed
        Af = A_sub.clone(); Af[i, j] = 0; Af[j, i] = 0
        A_fixed_A.append(float((reconverge(model, Af, X_proj_sub, Z_sub) - Z_sub).norm()))
        Ar = rescale_delete(A_sub, c_full, i, j)
        A_recomp_A.append(np.nan if Ar is None else
                          float((reconverge(model, Ar, X_proj_sub, Z_sub) - Z_sub).norm()))

        # (B) self-consistent base, subgraph-degree recomputed
        Bf = A0.clone(); Bf[i, j] = 0; Bf[j, i] = 0
        A_fixed_B.append(float((reconverge(model, Bf, X_proj_sub, Z0) - Z0).norm()))
        Araw2 = A_raw.clone(); Araw2[i, j] = 0; Araw2[j, i] = 0
        Br = normalize(Araw2)
        A_recomp_B.append(float((reconverge(model, Br, X_proj_sub, Z0) - Z0).norm()))

        min_deg_full.append(float(min(deg_full[i], deg_full[j])))
        min_deg_sub.append(float(min(deg_sub[i], deg_sub[j])))

    def rep(name, fx, rc):
        fx, rc = np.array(fx), np.array(rc)
        ok = ~np.isnan(rc)
        tau, _ = kendalltau(fx[ok], rc[ok])
        gap = np.mean(np.abs(fx[ok] - rc[ok]) / (np.abs(fx[ok]) + 1e-12))
        print(f"  {name}: tau={tau:.4f}  rel_gap={gap:.3f}")
        return fx, rc, ok

    print("\n(A) RESTRICTED base + full-graph-degree recomputed:")
    fxA, rcA, okA = rep("A", A_fixed_A, A_recomp_A)
    print("(B) SELF-CONSISTENT base + subgraph-degree recomputed:")
    fxB, rcB, okB = rep("B", A_fixed_B, A_recomp_B)

    # degree stratification (setup B, the clean one)
    md = np.array(min_deg_sub)
    relgap_B = np.abs(fxB - rcB) / (np.abs(fxB) + 1e-12)
    print("\nSetup B rel_gap vs min subgraph degree of the edge:")
    for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 99)]:
        m = (md >= lo) & (md <= hi)
        if m.sum():
            print(f"  deg in [{lo},{hi}]: n={int(m.sum())}  mean rel_gap={relgap_B[m].mean():.3f}")
    corr = np.corrcoef(md, relgap_B)[0, 1]
    print(f"  corr(min_deg, rel_gap) = {corr:.3f}   (expect negative: O(1/d))")


if __name__ == "__main__":
    main()
