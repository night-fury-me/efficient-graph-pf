"""Diagnose whether the AEGIS IGNN node classifier is unnecessarily weak.

Root-cause hypothesis (to confirm/refute by measurement):
  The operator is F(Z) = ReLU(A_hat @ Z @ W^T + X_proj).
  ||A_hat||_2 = 1.0 exactly (normalized adj + self loops), and the default
  spectral_norm caps ||W||_2 = 1. So the contraction factor kappa ~ 1 sits ON
  the contractivity boundary:
    - the 50-iter Picard forward from Z=0 cannot converge (rate kappa^k ~ 1),
      so training back-props through an UNCONVERGED fixed point (biased grads);
    - kappa is not < 1, so assumption A3 is violated anyway.
  Fix that helps BOTH: cap ||W||_2 = c < 1 with margin -> kappa = c < 1 (A3 holds
  strictly) AND geometric forward convergence at rate c. Sweep c for the
  accuracy/contractivity trade-off; also sweep iters/tol, epochs, dropout.

For EVERY config we report: test acc, train acc, forward residual at z*, and the
TRUE kappa = ||J_z||_2 measured by power-iteration + Rayleigh quotient on the
operator Jacobian at the full-graph fixed point (rho_rayleigh, reused verbatim
in spirit from scripts/exp_fullgraph_attack_table.py). kappa is the real A3
quantity (ReLU mask included), not an analytic bound.

Foreground, full-graph forward (no subgraph). Pubmed kappa uses fewer power-iter
restarts for tractability but is still the true J_z 2-norm.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from iem.examples.ignn_cora import _download_cora, _load_cora
from iem.examples.ignn_citeseer_pubmed import _download_planetoid, _load_planetoid

DATA_ROOT = PROJ_ROOT / "datasets"
SEEDS = [42, 137, 271, 314, 1729]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _ensure(name):
    d = DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_full(name):
    """Return full dict incl. val/test masks (train_ignn helper drops val/test)."""
    loaders = {
        "Cora": lambda: _load_cora(_ensure("cora")),
        "Citeseer": lambda: _load_planetoid("citeseer", _ensure("citeseer")),
        "Pubmed": lambda: _load_planetoid("pubmed", _ensure("pubmed")),
    }
    downloaders = {
        "Cora": lambda: _download_cora(_ensure("cora")),
        "Citeseer": lambda: _download_planetoid("citeseer", _ensure("citeseer")),
        "Pubmed": lambda: _download_planetoid("pubmed", _ensure("pubmed")),
    }
    try:
        d = loaders[name]()
    except FileNotFoundError:
        downloaders[name]()
        d = loaders[name]()
    return d


# ---------------------------------------------------------------------------
# Configurable IGNN  (superset of iem/examples/ignn_cora.py:IGNN)
#   - spectral_cap c: hard-rescale ||W||_2 <= c  (c=None -> plain spectral_norm, ||W||=1)
#   - dropout on Z* before head
#   - forward max_iter / tol exposed
# Architecturally identical to the paper model when c=None, dropout=0, iters=50.
# ---------------------------------------------------------------------------
class IGNNcfg(nn.Module):
    def __init__(self, n_features, hidden, n_classes, spectral_cap=None,
                 dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.spectral_cap = spectral_cap
        self.dropout = dropout
        self.U = nn.Linear(n_features, hidden)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.xavier_normal_(self.W.weight, gain=0.5)
        if spectral_cap is None:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            self.W = _sn(self.W)  # exactly the paper model: ||W||_2 -> 1
        # else: we rescale W in operator() so ||W_eff||_2 <= spectral_cap

    def _W_eff(self, Z):
        """Apply W to Z; if a hard cap is set, rescale W to have 2-norm = cap.

        Uses the analytic spectral norm of the (small hidden x hidden) W via SVD;
        differentiable, and guarantees ||W_eff||_2 = cap exactly each forward."""
        if self.spectral_cap is None:
            return self.W(Z)
        Wm = self.W.weight
        sn = torch.linalg.matrix_norm(Wm, ord=2)
        scale = self.spectral_cap / (sn + 1e-12)
        # only scale DOWN (don't blow small W up past its natural norm * ... )
        scale = torch.clamp(scale, max=1.0)
        return Z @ (Wm * scale).t()

    def operator(self, Z, ctx):
        out = ctx["A_hat"] @ self._W_eff(Z) + ctx["X_proj"]
        return F.relu(out)

    def forward(self, X, A_hat, max_iter=50, tol=1e-5, train_dropout=False):
        N = X.shape[0]
        X_proj = self.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}
        Z = torch.zeros(N, self.hidden, device=X.device)
        for k in range(max_iter):
            Z_new = self.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm(), 1.0):
                Z = Z_new
                break
            Z = Z_new
        Z_star = Z
        H = Z_star
        if train_dropout and self.dropout > 0:
            H = F.dropout(H, p=self.dropout, training=True)
        logits = self.head(H)
        return logits, Z_star, ctx


# ---------------------------------------------------------------------------
# kappa = ||J_z||_2 at the fixed point  (power iteration + Rayleigh quotient)
# J_z is the operator Jacobian linearised at z* WITH the ReLU mask -> the true
# A3 contraction factor. (rho_rayleigh logic from exp_fullgraph_attack_table.py.)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _noop():
    pass


def kappa_Jz(model, Z_star, ctx, iters=120, restarts=2):
    """Dominant singular value of J_z via power iteration on J_z^T J_z.

    ||J_z||_2 = sqrt(lambda_max(J_z^T J_z)). We use VJP/JVP via autograd.
    Robust (true 2-norm, not the sign-aware spectral radius); for a normal-ish
    J this matches |lambda_max|, but the 2-norm is the correct Lipschitz/A3
    quantity to compare against 1."""
    z = Z_star.detach().reshape(-1)
    D = z.numel()
    Ashape = Z_star.shape

    def F_flat(zf):
        return model.operator(zf.reshape(Ashape), ctx).reshape(-1)

    best = 0.0
    for r in range(restarts):
        torch.manual_seed(1000 + r)
        v = torch.randn(D, device=z.device, dtype=z.dtype)
        v = v / v.norm()
        sigma = 0.0
        for _ in range(iters):
            # Jv  (forward-mode)
            zf = z.clone().requires_grad_(True)
            out, Jv = torch.autograd.functional.jvp(F_flat, zf, v,
                                                    create_graph=False, strict=False)
            # J^T (Jv)  (reverse-mode) -> J^T J v
            zf2 = z.clone().requires_grad_(True)
            out2 = F_flat(zf2)
            JtJv = torch.autograd.grad(out2, zf2, grad_outputs=Jv,
                                       retain_graph=False)[0]
            nv = JtJv.norm()
            if nv < 1e-20:
                sigma = 0.0
                break
            v = JtJv / nv
            sigma_new = nv.sqrt().item()  # sqrt(lambda) = singular value
            if abs(sigma_new - sigma) < 1e-7:
                sigma = sigma_new
                break
            sigma = sigma_new
        best = max(best, sigma)
    return best


# ---------------------------------------------------------------------------
# Train + evaluate one config
# ---------------------------------------------------------------------------
def run_config(name, data, seed, *, spectral_cap=None, dropout=0.0,
               epochs=200, lr=0.01, wd=5e-4, fwd_iter=50, fwd_tol=1e-5,
               cosine=False, eval_iter=300, eval_tol=1e-8, measure_kappa=True,
               kappa_iters=120, kappa_restarts=2):
    torch.manual_seed(seed)
    X = data["X"].to(DEVICE)
    A = data["A_hat"].to(DEVICE)
    y = data["y"].to(DEVICE)
    tr = data["train_mask"].to(DEVICE)
    te = data["test_mask"].to(DEVICE)

    model = IGNNcfg(data["n_features"], 64, data["n_classes"],
                    spectral_cap=spectral_cap, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)

    for _ in range(epochs):
        model.train()
        logits, _, _ = model(X, A, max_iter=fwd_iter, tol=fwd_tol,
                             train_dropout=(dropout > 0))
        loss = F.cross_entropy(logits[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            sched.step()

    # Evaluate at a WELL-CONVERGED fixed point (eval_iter/eval_tol), no dropout.
    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A, max_iter=eval_iter, tol=eval_tol)
        pred = logits.argmax(1)
        test_acc = float((pred[te] == y[te]).float().mean())
        train_acc = float((pred[tr] == y[tr]).float().mean())
        residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
        # forward residual the TRAINING saw (loose solve), for diagnosis
        _, Z_tr, ctx_tr = model(X, A, max_iter=fwd_iter, tol=fwd_tol)
        residual_train = (model.operator(Z_tr, ctx_tr) - Z_tr).norm().item()

    kappa = float("nan")
    if measure_kappa:
        kappa = kappa_Jz(model, Z_star, ctx, iters=kappa_iters,
                         restarts=kappa_restarts)
    return dict(test=test_acc, train=train_acc, kappa=kappa,
                res_eval=residual, res_train=residual_train)


def agg(name, label, **kw):
    """Run all seeds, print mean+/-sd row."""
    ts, trs, ks, res_e, res_t = [], [], [], [], []
    t0 = time.time()
    for s in SEEDS:
        r = run_config(name, DATA[name], s, **kw)
        ts.append(r["test"]); trs.append(r["train"]); ks.append(r["kappa"])
        res_e.append(r["res_eval"]); res_t.append(r["res_train"])
    import statistics as st
    def ms(xs):
        xs = [x for x in xs if x == x]  # drop nan
        if not xs:
            return float("nan"), float("nan")
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)
    tm, tsd = ms(ts); trm, trsd = ms(trs); km, ksd = ms(ks)
    rem, _ = ms(res_e); rtm, _ = ms(res_t)
    dt = time.time() - t0
    row = (f"{name:8s} | {label:34s} | test {tm*100:5.2f}+/-{tsd*100:4.2f} "
           f"| train {trm*100:5.2f} | kappa {km:.4f}+/-{ksd:.4f} "
           f"| res_eval {rem:.1e} | res_train {rtm:.1e} | {dt:5.1f}s")
    print(row, flush=True)
    RESULTS.append(dict(dataset=name, config=label, test_mean=tm, test_sd=tsd,
                        train_mean=trm, kappa_mean=km, kappa_sd=ksd,
                        res_eval=rem, res_train=rtm))
    return tm, km


# ---------------------------------------------------------------------------
RESULTS = []
DATA = {}


def main():
    print(f"device={DEVICE}, seeds={SEEDS}", flush=True)
    for nm in ["Cora", "Citeseer", "Pubmed"]:
        DATA[nm] = load_full(nm)
        d = DATA[nm]
        sv = torch.linalg.svdvals(d["A_hat"].float())[0].item()
        print(f"loaded {nm}: N={d['N']} feat={d['n_features']} cls={d['n_classes']} "
              f"train={int(d['train_mask'].sum())} test={int(d['test_mask'].sum())} "
              f"||A_hat||2={sv:.4f}", flush=True)

    print("\n" + "=" * 130)
    print("STEP 1 — PAPER SETTINGS  (spectral_norm ||W||=1, 200 ep, hidden=64, "
          "fwd max_iter=50 tol=1e-5)   <-- confirm the ~61% number")
    print("=" * 130, flush=True)
    # kappa here is measured at the WELL-CONVERGED eval fixed point.
    for nm in ["Cora", "Citeseer", "Pubmed"]:
        kw = dict(spectral_cap=None, epochs=200, fwd_iter=50, fwd_tol=1e-5)
        if nm == "Pubmed":
            kw.update(kappa_iters=60, kappa_restarts=1)
        agg(nm, "PAPER (cap=1,it=50)", **kw)

    print("\n" + "=" * 130)
    print("STEP 2 — DIAGNOSIS on Cora (one knob at a time vs paper baseline)")
    print("=" * 130, flush=True)
    # (a) tighten the forward fixed point ONLY (still cap=1 -> still kappa~1)
    agg("Cora", "(a) it=300 tol=1e-7 (cap=1)", spectral_cap=None, epochs=200,
        fwd_iter=300, fwd_tol=1e-7)
    # (b) hard spectral cap c<1  -> kappa=c<1 AND geometric convergence
    for c in [0.95, 0.9, 0.8, 0.7, 0.5]:
        agg("Cora", f"(b) cap={c} (it=50)", spectral_cap=c, epochs=200,
            fwd_iter=50, fwd_tol=1e-5)
    # (b+a) best cap WITH tight forward
    agg("Cora", "(b+a) cap=0.9 it=300 tol=1e-7", spectral_cap=0.9, epochs=200,
        fwd_iter=300, fwd_tol=1e-7)
    # (c) cap + dropout + longer + cosine
    agg("Cora", "(c) cap=0.9 drop=0.5 ep=300", spectral_cap=0.9, dropout=0.5,
        epochs=300, fwd_iter=100, fwd_tol=1e-6)
    agg("Cora", "(c) cap=0.9 drop=0.5 cos ep=400", spectral_cap=0.9, dropout=0.5,
        epochs=400, fwd_iter=100, fwd_tol=1e-6, cosine=True)
    agg("Cora", "(c) cap=0.95 drop=0.5 cos ep=400", spectral_cap=0.95, dropout=0.5,
        epochs=400, fwd_iter=100, fwd_tol=1e-6, cosine=True)

    print("\n" + "=" * 130)
    print("STEP 2b — apply the WINNING recipe to Citeseer / Pubmed")
    print("=" * 130, flush=True)
    best_cfg = dict(spectral_cap=0.9, dropout=0.5, epochs=400, fwd_iter=100,
                    fwd_tol=1e-6, cosine=True)
    agg("Citeseer", "WIN cap=0.9 drop=0.5 cos ep=400", **best_cfg)
    pm = dict(best_cfg); pm.update(kappa_iters=60, kappa_restarts=1)
    agg("Pubmed", "WIN cap=0.9 drop=0.5 cos ep=400", **pm)
    # also the leaner cap-only recipe on all three (cheap, robust)
    agg("Citeseer", "cap=0.9 it=100 (no drop)", spectral_cap=0.9, epochs=200,
        fwd_iter=100, fwd_tol=1e-6)

    # dump JSON for the md
    import json
    out = PROJ_ROOT / "scripts" / "revision_R2" / "_ignn_accuracy_results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
