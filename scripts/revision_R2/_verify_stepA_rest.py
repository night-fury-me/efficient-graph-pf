"""Remaining Step-A verification: Pubmed accuracy/kappa (5 seeds) + sensitivity
pipeline on a fresh c=0.9 Cora model. Split out so each chunk finishes well
inside a single foreground window.
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
from scripts.revision_R2.ignn_accuracy_diag import load_full  # noqa: E402
from scripts.revision_R2._verify_stepA import kappa_Jz, build_op  # noqa: E402
from iem.scalable import ScalableSensitivity  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 137, 271, 314, 1729]


def main():
    print(f"device={DEVICE}  seeds={SEEDS}", flush=True)
    print("=" * 96)
    print("PART (a)+(b) cont'd: PUBMED accuracy (5 seeds) + kappa=||J_z||_2  (<1)")
    print("=" * 96, flush=True)
    d = load_full("Pubmed")
    X = d["X"].to(DEVICE); A = d["A_hat"].to(DEVICE); y = d["y"].to(DEVICE)
    tr = d["train_mask"].to(DEVICE); te = d["test_mask"].to(DEVICE)
    sv = torch.linalg.svdvals(A.float())[0].item()
    accs, kappas = [], []
    t0 = time.time()
    for s in SEEDS:
        model = train_ignn(X, A, y, tr, d["n_features"], d["n_classes"], DEVICE, s)
        model.eval()
        with torch.no_grad():
            logits, Z_star, ctx = model(X, A, max_iter=300, tol=1e-8)
            acc = float((logits.argmax(1)[te] == y[te]).float().mean())
        k = kappa_Jz(model, Z_star, ctx, iters=60, restarts=1)
        accs.append(acc); kappas.append(k)
        print(f"  Pubmed    seed={s:4d}  test={acc*100:5.2f}%  kappa={k:.4f}", flush=True)
    am, asd = st.mean(accs), st.stdev(accs)
    km, ksd = st.mean(kappas), st.stdev(kappas)
    print(f"  >> Pubmed: test {am*100:5.2f} +/- {asd*100:4.2f}  | kappa {km:.4f} "
          f"+/- {ksd:.4f} (max {max(kappas):.4f})  | ||A_hat||2={sv:.4f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 96)
    print("PART (c): SENSITIVITY PIPELINE on a fresh c=0.9 Cora model (build_op path)")
    print("=" * 96, flush=True)
    dc = load_full("Cora")
    Xc = dc["X"].to(DEVICE); Ac = dc["A_hat"].to(DEVICE); yc = dc["y"].to(DEVICE)
    trc = dc["train_mask"].to(DEVICE)
    model = train_ignn(Xc, Ac, yc, trc, dc["n_features"], dc["n_classes"], DEVICE, 42)
    t1 = time.time()
    op, Z_star, ctx, rho, rebuilt = build_op(model, Xc, Ac)
    print(f"  ScalableSensitivity built: N={op.N} edges={op.num_edges} "
          f"rho(J_z)={rho:.4f} rebuilt={rebuilt}  [{time.time()-t1:.0f}s]", flush=True)
    vij = []
    for k in range(min(5, op.num_edges)):
        val = float(op._column(k).norm().item())
        vij.append(val)
        i, j = op.edge_list[k]
        print(f"  v_ij[edge {k}=({i},{j})] = ||S_c[:,k]|| = {val:.6e}", flush=True)
    _U, sigma, _Vh = op.top_k_svd(k=1)
    sig1 = float(sigma[0])
    print(f"  sigma_1(S_c) [matrix-free] = {sig1:.6e}", flush=True)
    finite = all(v == v and v != float("inf") for v in vij) and (sig1 == sig1) \
        and sig1 != float("inf")
    print(f"  ALL FINITE/SENSIBLE = {finite}", flush=True)

    a_ok = 76 <= am * 100 <= 81 and asd * 100 < 2.0
    k_ok = max(kappas) < 1.0
    print("\nSUMMARY (rest):")
    print(f"  Pubmed acc={am*100:.2f}+/-{asd*100:.2f} (expect 76-81,sd<2) "
          f"{'OK' if a_ok else 'CHECK'}  | kappa_max={max(kappas):.4f} "
          f"{'OK <1' if k_ok else 'FAIL'}")
    print(f"  Sensitivity finite: {'PASS' if finite else 'FAIL'}  "
          f"sigma1={sig1:.4f} rho={rho:.4f}")

    import json
    out = PROJ_ROOT / "scripts" / "revision_R2" / "_verify_stepA_rest_results.json"
    out.write_text(json.dumps(dict(
        pubmed_acc=am * 100, pubmed_acc_sd=asd * 100, pubmed_kappa_max=max(kappas),
        pubmed_kappa_mean=km, A_norm=sv,
        sens_rho=rho, sens_rebuilt=rebuilt, sens_sigma1=sig1, sens_vij=vij,
        sens_finite=finite), indent=2))
    print(f"wrote {out}", flush=True)
    print("REST_DONE", flush=True)


if __name__ == "__main__":
    main()
