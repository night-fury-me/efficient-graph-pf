"""Independent confirmation of the IGNN accuracy fix + kappa reporting resolution.

(1) REAL paper model: train iem.examples.ignn_cora.IGNN via _common.train_ignn
    (NOT the harness reimplementation) on Cora/Citeseer; measure test acc AND BOTH
    rho(J_z) (spectral radius, rho_rayleigh) and ||J_z||_2 (operator norm, kappa_Jz).
    -> resolves whether the paper's reported "kappa=0.14-0.59" is rho or ||J_z||_2,
       and confirms the ~58-61% weak-accuracy baseline on the actual model.
(2) cap=0.9 winning recipe (harness run_config) on Cora -> confirm ~80% at kappa<1.
"""
import sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.revision_R2._common import train_ignn
from scripts.revision_R2.ignn_accuracy_diag import kappa_Jz, load_full, run_config
from scripts.exp_fullgraph_attack_table import rho_rayleigh, build_op

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 137]
print(f"device={dev} seeds={SEEDS}", flush=True)

print("\n=== (1) REAL paper IGNN (cap=1, 200ep): acc + rho(J_z) + ||J_z||_2 ===", flush=True)
for ds in ["Cora", "Citeseer"]:
    d = load_full(ds)
    X, A, y = d["X"].to(dev), d["A_hat"].to(dev), d["y"].to(dev)
    tr, te = d["train_mask"].to(dev), d["test_mask"].to(dev)
    accs, rhos, norms = [], [], []
    for s in SEEDS:
        m = train_ignn(X, A, y, tr, d["n_features"], d["n_classes"], dev, s)
        m.eval()
        with torch.no_grad():
            _, Z_star, ctx = m(X, A)
            logits = m.head(Z_star)
            accs.append(float((logits.argmax(1)[te] == y[te]).float().mean()))
        # rho(J_z): spectral radius via Rayleigh power iteration
        op, _, _, rho, _ = build_op(m, X, A)
        rhos.append(rho)
        del op
        # ||J_z||_2: operator 2-norm via power iteration on J^T J
        norms.append(kappa_Jz(m, Z_star, ctx, iters=120, restarts=2))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"{ds:9s}: test_acc={st.mean(accs)*100:5.1f}%  rho(J_z)={st.mean(rhos):.3f}  "
          f"||J_z||_2={st.mean(norms):.3f}  (paper claims kappa=0.14-0.59)", flush=True)

print("\n=== (2) cap=0.9 winning recipe on Cora (confirm ~80%, kappa<1) ===", flush=True)
D = load_full("Cora")
for s in SEEDS:
    r = run_config("Cora", D, s, spectral_cap=0.9, dropout=0.5, epochs=400,
                   fwd_iter=100, fwd_tol=1e-6, cosine=True)
    print(f"  seed{s}: test={r['test']*100:5.1f}%  train={r['train']*100:5.1f}%  "
          f"kappa(||J_z||_2)={r['kappa']:.3f}  res_eval={r['res_eval']:.1e}", flush=True)
print("\nCONFIRM DONE", flush=True)
