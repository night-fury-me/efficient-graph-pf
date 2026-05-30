"""X9 -- Earn the phase transition: *exhibit* the resolvent blow-up.

Theorem 1(b) (corrected, item T1) gives the unconditional bound
    ||(I - J_z)^{-1}||_2 >= 1 / dist(1, spec(J_z)) = 1 / min_i |1 - lambda_i(J_z)|,
with divergence Omega(1/(eps_crit - eps)) when a REAL eigenvalue of J_z -> +1.

The trained ReLU model never reaches this cliff: the ReLU active-set mask sparsifies
J_z so its largest eigenvalue stays <= ~0.42 even at the spectral cap kappa=0.99
(curve ii -- the 2-4x safety margin, also in results/exp_phase_transition.csv). To
*demonstrate* the predicted blow-up (curve i) we scale a model-derived operator toward
the contraction boundary, driving a REAL eigenvalue lambda -> +1.

Why the symmetric part (critique-driven design choice):
  The linear Jacobian J_lin = A (x) W is non-normal -- W (trained 64x64) has an
  all-complex spectrum, so radially scaling J_lin sweeps eigenvalues *past* the line
  Re=1 but never *through the point* 1; dist(1,spec) bottoms out at |Im| and the
  resolvent merely PEAKS, it does not diverge. To exhibit a genuine real eigenvalue
  -> +1 we use the symmetric part  S = A (x) W_sym,  W_sym = (W + W^T)/2  (symmetric =>
  real spectrum). This is exactly the real-axis contraction margin (numerical abscissa)
  of J_z; scaling lambda_max(S) -> 1 gives the exact identity ||(I - S)^{-1}|| = 1/(1-s).

Two curves on the (dist(1,spec), resolvent) / (lambda, resolvent) plane:
  (i)  STRESS  : S scaled so lambda_max -> s -> 1;  resolvent = 1/(1-s)  (the cliff).
  (ii) TRAINED : actual masked J_z at kappa=0.99;  lambda bounded away from 1, flat resolvent.

Correctness gate (critique step, built in): the analytic Jacobians
  J_lin    = A (x) W            (row-major vec convention)
  J_masked = diag(relu') . J_lin
are asserted equal to the trusted autograd `compute_jacobian` on the first seed.

Dataset: Cora (50-node BFS subgraph). Seeds: 10. Heavy linalg in float64 on CPU.

Outputs:
  results/exp_phase_transition_stress.csv

Usage:
    .venv/bin/python scripts/exp_phase_transition_stress.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import extract_ego_subgraph
from iem.examples.ignn_cora import _load_cora
from iem.ift import compute_jacobian
# Reuse the kappa-controlled model + trainer from the safe-side experiment so
# curve (i) and curve (ii) come from identically-trained models.
from exp_phase_transition import IGNN_Kappa, SEEDS, set_seed, train_ignn_kappa  # noqa: E402

KAPPA = 0.99                      # worst-case spectral cap (closest to the boundary)
STRESS_GRID = [0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
SUBGRAPH_NODES = 50
VERIFY_TOL = 1e-3                 # float32-autograd vs float64-analytic agreement


def resolvent_norm_2(J: torch.Tensor) -> float:
    """||(I - J)^{-1}||_2 = 1 / sigma_min(I - J). No inversion -> stable near the cliff."""
    D = J.shape[0]
    I = torch.eye(D, dtype=J.dtype, device=J.device)
    smin = float(torch.linalg.svdvals(I - J).min())
    return float("inf") if smin < 1e-300 else 1.0 / smin


def run_seed(data, device, seed, verify: bool):
    set_seed(seed)
    model, Z_star, ctx, _ = train_ignn_kappa(data, device, seed, KAPPA)
    A_hat = data["A_hat"].to(device)

    # 50-node BFS subgraph (same convention as the safe-side experiment)
    idx = extract_ego_subgraph(A_hat, max_nodes=SUBGRAPH_NODES)
    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    # reconverge the equilibrium on the subgraph
    Z = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(500):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-8:
                break
            Z = Z_new
    Z_sub = Z_new

    W = model.W.weight.detach()                                 # (h, h)
    pre = A_sub @ (Z_sub @ W.T) + X_proj_sub                    # pre-activation at eq. (N,h)
    mask = (pre > 0).reshape(-1)                                # ReLU' mask, row-major vec

    # --- analytic Jacobians (float64 on CPU for robust eig/svd) ---
    A64 = A_sub.detach().double().cpu()
    A64 = 0.5 * (A64 + A64.T)                                  # enforce exact symmetry (A_hat is symmetric)
    W64 = W.double().cpu()
    Wsym64 = 0.5 * (W64 + W64.T)
    m64 = mask.double().cpu()
    J_lin = torch.kron(A64, W64)                                # vec(AZW^T) = (A x W) vec(Z), row-major
    J_masked = m64[:, None] * J_lin                            # diag(mask) . J_lin
    S_lin = torch.kron(A64, Wsym64)                            # symmetric part of J_lin (A symmetric)

    # --- correctness gate: analytic vs trusted autograd (first seed only) ---
    verify_info = {}
    if verify:
        def F_lin(z):
            Zz = z.reshape(Z_sub.shape)
            return (A_sub @ (Zz @ W.T) + X_proj_sub).reshape(-1)

        def F_relu(z):
            return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)

        J_lin_ad = compute_jacobian(F_lin, Z_sub).double().cpu()
        J_msk_ad = compute_jacobian(F_relu, Z_sub).double().cpu()
        verify_info = {
            "lin_max_abs_err": float((J_lin - J_lin_ad).abs().max()),
            "masked_max_abs_err": float((J_masked - J_msk_ad).abs().max()),
            "sym_check": float((S_lin - 0.5 * (J_lin + J_lin.T)).abs().max()),  # ==0 by construction
        }

    # --- curve (ii): trained / masked safe point (actual, non-normal J_z) ---
    ev_m = torch.linalg.eigvals(J_masked)                       # complex spectrum
    num_abscissa = float(torch.linalg.eigvalsh(0.5 * (J_masked + J_masked.T)).max())
    safe = {
        "lambda_real_max": float(ev_m.real.max()),
        "rho": float(ev_m.abs().max()),
        "num_abscissa": num_abscissa,                           # lambda_max(sym(J_z)) = real-axis margin
        "dist1": float((ev_m - 1.0).abs().min()),
        "resolvent": resolvent_norm_2(J_masked),
    }

    # --- curve (i): symmetric-part stress, real eigenvalue -> +1 ---
    nu = torch.linalg.eigvalsh(S_lin)                           # real eigenvalues of A (x) W_sym
    mu1 = float(nu.max())                                       # top real eigenvalue (Perron-of-sym)
    assert mu1 > 1e-8, f"symmetric part has no positive eigenvalue (mu1={mu1}); stress invalid"
    rho_lin = float(torch.linalg.eigvals(J_lin).abs().max())    # spectral radius of full (non-normal) op
    sweep = []
    for s in STRESS_GRID:
        ev_s = nu * (s / mu1)                                   # scaled real spectrum; top -> s
        dist1 = float((ev_s - 1.0).abs().min())                # = 1 - s (top eigenvalue closest to 1)
        sweep.append({
            "s": s,
            "lambda_top": float(ev_s.max()),                    # the real eigenvalue -> 1
            "dist1": dist1,
            "resolvent": float("inf") if dist1 < 1e-300 else 1.0 / dist1,  # exact for symmetric
        })

    return {"seed": seed, "safe": safe, "mu1": mu1, "rho_lin": rho_lin,
            "sweep": sweep, "verify": verify_info}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f"  GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else ""))

    data = _load_cora(Path("datasets/cora"))
    print(f"Cora: N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")
    print(f"kappa={KAPPA}, subgraph={SUBGRAPH_NODES} nodes, seeds={len(SEEDS)}", flush=True)

    rows, t0 = [], time.time()
    for si, seed in enumerate(SEEDS):
        print(f"\n[{si+1}/{len(SEEDS)}] seed={seed} ...", flush=True)
        r = run_seed(data, device, seed, verify=(si == 0))

        if r["verify"]:
            v = r["verify"]
            print(f"    VERIFY lin_err={v['lin_max_abs_err']:.2e}  "
                  f"masked_err={v['masked_max_abs_err']:.2e}  "
                  f"sym_check={v['sym_check']:.2e}  (tol={VERIFY_TOL:.0e})", flush=True)
            assert v["lin_max_abs_err"] < VERIFY_TOL, "linear Jacobian convention mismatch!"
            assert v["masked_max_abs_err"] < VERIFY_TOL, "masked Jacobian / mask mismatch!"
            assert v["sym_check"] < 1e-9, "symmetric-part construction mismatch!"
            print("    VERIFY PASSED.", flush=True)

        s = r["safe"]
        print(f"    TRAINED (masked): lambda_real_max={s['lambda_real_max']:.4f}  "
              f"num_abscissa={s['num_abscissa']:.4f}  dist1={s['dist1']:.4f}  "
              f"resolvent={s['resolvent']:.3f}", flush=True)
        print(f"    STRESS base mu1(sym)={r['mu1']:.4f}  rho_lin(full)={r['rho_lin']:.4f}", flush=True)
        for w in r["sweep"]:
            print(f"      s={w['s']:.4f}  lambda_top={w['lambda_top']:.4f}  "
                  f"dist1={w['dist1']:.3e}  resolvent={w['resolvent']:.3e}", flush=True)

        rows.append({
            "curve": "trained", "seed": seed, "s": "", "kappa": KAPPA,
            "lambda": s["lambda_real_max"], "num_abscissa": s["num_abscissa"],
            "dist1": s["dist1"], "resolvent": s["resolvent"],
            "mu1": r["mu1"], "rho_lin": r["rho_lin"],
        })
        for w in r["sweep"]:
            rows.append({
                "curve": "stress", "seed": seed, "s": w["s"], "kappa": KAPPA,
                "lambda": w["lambda_top"], "num_abscissa": "",
                "dist1": w["dist1"], "resolvent": w["resolvent"],
                "mu1": r["mu1"], "rho_lin": r["rho_lin"],
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

    out = Path("results/exp_phase_transition_stress.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "curve", "seed", "s", "kappa", "lambda", "num_abscissa",
            "dist1", "resolvent", "mu1", "rho_lin",
        ])
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {out}")

    def agg(curve, key, sval=None):
        vals = [r[key] for r in rows if r["curve"] == curve
                and (sval is None or r["s"] == sval)
                and isinstance(r[key], float) and np.isfinite(r[key])]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    print("\n" + "=" * 78)
    am, asd = agg("trained", "num_abscissa")
    rm, rsd = agg("trained", "resolvent")
    dm, dsd = agg("trained", "dist1")
    print(f"TRAINED (masked J_z, kappa=0.99):  num_abscissa(lambda)={am:.3f}+/-{asd:.3f}  "
          f"dist1={dm:.3f}+/-{dsd:.3f}  resolvent={rm:.3f}+/-{rsd:.3f}")
    print("STRESS (symmetric part, lambda -> 1):")
    print(f"  {'s(lambda)':>10} {'dist1=1-s':>12} {'resolvent (mean+/-sd)':>26}")
    for s in STRESS_GRID:
        rm, rsd = agg("stress", "resolvent", s)
        print(f"  {s:>10.4f} {1-s:>12.4e} {rm:>14.2f} +/- {rsd:<8.2f}")
    print(f"\nSafety margin: trained sits at lambda~{am:.2f} (dist1~{dm:.2f}); the cliff is lambda->1.")
    print("Expected: stress resolvent = 1/(1-s) diverges; trained resolvent flat (~1.2-1.8).")


if __name__ == "__main__":
    sys.exit(main() or 0)
