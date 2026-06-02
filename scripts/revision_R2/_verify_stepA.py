"""Step-A verification: confirm the c=0.9 IGNN recipe is the repo default and
that accuracy / contractivity / the sensitivity pipeline all hold.

Parts:
  (a) ACCURACY (5 seeds) via the updated _common.train_ignn on the public split.
  (b) CONTRACTIVITY kappa = ||J_z||_2 (power iteration on J^T J at the fixed
      point) per dataset -- must be < 1.
  (c) SENSITIVITY PIPELINE: build ScalableSensitivity on a fresh c=0.9 Cora model
      (same build_op path as exp_fullgraph_attack_table) and compute a few v_ij
      and sigma_1(S_c); confirm finite/sensible.

Run foreground. Reports real numbers.
"""
from __future__ import annotations

import sys
import time
import statistics as st
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from scripts.revision_R2._common import train_ignn  # noqa: E402
# load_full (full dict incl. val/test masks; load_dataset only returns train_mask)
from scripts.revision_R2.ignn_accuracy_diag import load_full  # noqa: E402
from iem.scalable import ScalableSensitivity  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 137, 271, 314, 1729]


# ---------------------------------------------------------------------------
# kappa = ||J_z||_2 at the fixed point via power iteration on J^T J.
# J_z is the operator Jacobian linearised at z* WITH the ReLU mask (the true A3
# contraction factor). Uses autograd jvp/vjp through model.operator -- i.e. it
# differentiates the ACTUAL (capped) operator, proving the cap is transparent.
# ---------------------------------------------------------------------------
def kappa_Jz(model, Z_star, ctx, iters=120, restarts=2):
    N, H = Z_star.shape
    Zc = Z_star.detach().requires_grad_(True)
    f0 = model.operator(Zc, ctx)

    def jvp(v):  # J_z @ v  (forward-mode via double-backward trick)
        u = torch.zeros_like(f0, requires_grad=True)
        (g,) = torch.autograd.grad(f0, Zc, grad_outputs=u, create_graph=True)
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v, retain_graph=True)
        return jv

    def vjp(v):  # J_z^T @ v
        (g,) = torch.autograd.grad(f0, Zc, grad_outputs=v, retain_graph=True)
        return g

    best = 0.0
    for _ in range(restarts):
        v = torch.randn(N, H, device=Z_star.device)
        v = v / v.norm()
        lam = 0.0
        for _ in range(iters):
            w = vjp(jvp(v))           # (J^T J) v
            lam = w.norm().item()
            nv = w.norm()
            if nv < 1e-30:
                break
            v = w / nv
        best = max(best, lam ** 0.5)  # sigma_1 = sqrt(lambda_max(J^T J))
    return best


# ---------------------------------------------------------------------------
# build_op: same path as exp_fullgraph_attack_table.build_op (matrix-free op +
# trustworthy rho via Rayleigh power-iteration; rebuild with more Neumann terms
# if rho is near the contractivity boundary).
# ---------------------------------------------------------------------------
def rho_rayleigh(op, iters=200):
    # _jvp_Jz operates on a FLAT (N*H,) vector (z_star-shaped after reshape).
    v = torch.randn(op.z_star.numel(), device=op.z_star.device, dtype=op.z_star.dtype)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        Jv = op._jvp_Jz(v)
        lam = float((v * Jv).sum().item())   # Rayleigh quotient (real dominant eig)
        nv = Jv.norm()
        if nv < 1e-30:
            break
        v = Jv / nv
    return lam


def build_op(model, X, A, rho_thresh=0.98):
    with torch.no_grad():
        _, Z_star, ctx = model(X, A)

    def F_op(z, c):
        return model.operator(z, c)

    op = ScalableSensitivity(F_op, Z_star, ctx)
    try:
        rho = rho_rayleigh(op)
    except Exception:
        rho = float(op._estimate_rho()) if hasattr(op, "_estimate_rho") else float("nan")
    rebuilt = False
    if rho == rho and rho >= rho_thresh:
        op = ScalableSensitivity(F_op, Z_star, ctx, neumann_terms=3000)
        rebuilt = True
    return op, Z_star, ctx, rho, rebuilt


def main():
    print(f"device={DEVICE}  seeds={SEEDS}", flush=True)
    print("=" * 96)
    print("PART (a)+(b): ACCURACY (5 seeds) + CONTRACTIVITY kappa=||J_z||_2  (must be <1)")
    print("=" * 96, flush=True)

    summary = {}
    for nm in ["Cora", "Citeseer", "Pubmed"]:
        d = load_full(nm)
        X = d["X"].to(DEVICE); A = d["A_hat"].to(DEVICE); y = d["y"].to(DEVICE)
        tr = d["train_mask"].to(DEVICE); te = d["test_mask"].to(DEVICE)
        sv = torch.linalg.svdvals(A.float())[0].item()
        accs, kappas = [], []
        ki, kr = (60, 1) if nm == "Pubmed" else (120, 2)
        t0 = time.time()
        for s in SEEDS:
            model = train_ignn(X, A, y, tr, d["n_features"], d["n_classes"],
                               DEVICE, s)  # default recipe: c=0.9, drop=0.5, 400ep
            model.eval()
            with torch.no_grad():
                logits, Z_star, ctx = model(X, A, max_iter=300, tol=1e-8)
                pred = logits.argmax(1)
                acc = float((pred[te] == y[te]).float().mean())
            k = kappa_Jz(model, Z_star, ctx, iters=ki, restarts=kr)
            accs.append(acc); kappas.append(k)
            print(f"  {nm:9s} seed={s:4d}  test={acc*100:5.2f}%  kappa={k:.4f}",
                  flush=True)
        am, asd = st.mean(accs), (st.stdev(accs) if len(accs) > 1 else 0.0)
        km, ksd = st.mean(kappas), (st.stdev(kappas) if len(kappas) > 1 else 0.0)
        kmax = max(kappas)
        summary[nm] = dict(acc=am * 100, acc_sd=asd * 100, kappa=km, kappa_sd=ksd,
                           kappa_max=kmax, A_norm=sv, dt=time.time() - t0)
        print(f"  >> {nm}: test {am*100:5.2f} +/- {asd*100:4.2f}  | "
              f"kappa {km:.4f} +/- {ksd:.4f} (max {kmax:.4f})  | ||A_hat||2={sv:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 96)
    print("PART (c): SENSITIVITY PIPELINE on a fresh c=0.9 Cora model (build_op path)")
    print("=" * 96, flush=True)
    d = load_full("Cora")
    X = d["X"].to(DEVICE); A = d["A_hat"].to(DEVICE); y = d["y"].to(DEVICE)
    tr = d["train_mask"].to(DEVICE)
    model = train_ignn(X, A, y, tr, d["n_features"], d["n_classes"], DEVICE, 42)
    op, Z_star, ctx, rho, rebuilt = build_op(model, X, A)
    print(f"  ScalableSensitivity built: N={op.N} edges={op.num_edges} "
          f"rho(J_z)={rho:.4f} rebuilt={rebuilt}", flush=True)
    # a few v_ij (edge vulnerabilities) -- ||S_c[:,k]|| via _column (matrix-free)
    ks = list(range(min(5, op.num_edges)))
    vij = []
    for k in ks:
        try:
            val = float(op._column(k).norm().item())
        except Exception as e:
            val = float("nan")
            print(f"    _column({k}) raised {type(e).__name__}: {e}", flush=True)
        vij.append(val)
        i, j = op.edge_list[k]
        print(f"  v_ij[edge {k}=({i},{j})] = ||S_c[:,k]|| = {val:.6e}", flush=True)
    # sigma_1(S_c) matrix-free (randomized SVD: returns U, sigma, Vh)
    try:
        _U, sigma, _Vh = op.top_k_svd(k=1)
        sig1 = float(sigma[0])
    except Exception as e:
        sig1 = float("nan")
        print(f"  top_k_svd raised {type(e).__name__}: {e}", flush=True)
    print(f"  sigma_1(S_c) [matrix-free] = {sig1:.6e}", flush=True)
    finite = all(v == v for v in vij) and (sig1 == sig1) and sig1 != float("inf")
    print(f"  ALL FINITE/SENSIBLE = {finite}", flush=True)

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    ok_acc, ok_kappa = True, True
    bars = {"Cora": (78, 83), "Citeseer": (67, 72), "Pubmed": (76, 81)}
    for nm, sdic in summary.items():
        lo, hi = bars[nm]
        a_ok = lo <= sdic["acc"] <= hi and sdic["acc_sd"] < 2.0
        k_ok = sdic["kappa_max"] < 1.0
        ok_acc &= a_ok; ok_kappa &= k_ok
        print(f"  {nm:9s} acc={sdic['acc']:5.2f}+/-{sdic['acc_sd']:4.2f} "
              f"(expect {lo}-{hi}, sd<2) {'OK' if a_ok else 'CHECK'}  | "
              f"kappa_max={sdic['kappa_max']:.4f} {'OK <1' if k_ok else 'FAIL >=1'}")
    print(f"\n  ACCURACY BAR: {'PASS' if ok_acc else 'REVIEW'}   "
          f"CONTRACTIVITY BAR: {'PASS' if ok_kappa else 'FAIL'}   "
          f"SENSITIVITY FINITE: {'PASS' if finite else 'FAIL'}")

    import json
    out = PROJ_ROOT / "scripts" / "revision_R2" / "_verify_stepA_results.json"
    out.write_text(json.dumps({k: v for k, v in summary.items()} |
                              {"_sensitivity": {"rho": rho, "rebuilt": rebuilt,
                                                "sigma1_Sc": sig1, "v_ij": vij,
                                                "finite": finite}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
