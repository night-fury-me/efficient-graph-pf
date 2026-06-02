#!/usr/bin/env python3
r"""FEASIBILITY PROBE (AEGIS experiment #3): does AEGIS's matrix-free
structural-sensitivity machinery (Neumann resolvent + power-iteration sigma_1)
work on a *monotone* graph equilibrium model (MonDEQ-style), which is
structurally different from the paper's spectral-cap IGNN?

THE CRITICAL CONCERN
--------------------
AEGIS builds  S_c = (I - J_z)^{-1} J_A P_c  and inverts (I - J_z) with a
TRUNCATED NEUMANN series  sum_{k<K} J_z^k, which converges iff the SPECTRAL
RADIUS rho(J_z) < 1.

  * The IGNN guarantees this by a spectral-NORM cap: ||W||_2 <= 0.9 and
    ||A_hat||_2 = 1  =>  ||J_z||_2 < 1  =>  rho(J_z) <= ||J_z||_2 < 1.

  * A MonDEQ instead guarantees a UNIQUE equilibrium by MONOTONICITY: the
    Winston-Kolter parameterization makes  I - J_z  m-strongly monotone, i.e.
    sym(I - J_z) = (1/2)((I-J_z)+(I-J_z)^T) >= m I  (m>0).  Strong monotonicity
    bounds only the SYMMETRIC part: it forces  Re(lambda(J_z)) <= 1 - m < 1,
    but it does NOT bound |lambda(J_z)|.  A non-normal J_z can have eigenvalues
    with small/negative real part but large MODULUS, so rho(J_z) can be >= 1.
    If so, AEGIS's Neumann series DIVERGES and S_c cannot be formed
    matrix-free -- a structural incompatibility.

This probe answers EMPIRICALLY whether that happens on a trained graph MonDEQ.

THE MONOTONE GRAPH DEQ WE BUILD (and EXACTLY how it differs from the IGNN)
-------------------------------------------------------------------------
Equilibrium (same algebraic form as the IGNN so the comparison is fair):
        Z* = sigma( A_hat @ Z* @ W^T + X @ U )          (sigma = ReLU)
flattened  z = vec(Z),  so the operator linearization at z* is
        J_z = diag(sigma'(.)) @ (W (x) A_hat)            (Kronecker; A_hat sym).

DIFFERENCE 1 -- the parameterization of W (contraction MECHANISM):
  IGNN:   hard spectral-norm cap  ||W||_2 <= c=0.9  (rescale W down each fwd).
  MonDEQ: Winston-Kolter MONOTONE parameterization on the channel matrix
            W = (1 - m) I - A_par^T A_par + (B_par - B_par^T),
          so sym(W) = (1-m)I - A_par^T A_par <= (1-m) I  (PSD subtraction).
          This makes  I - W  m-strongly monotone IN CHANNEL SPACE.  There is
          NO 2-norm cap on W: ||W||_2 is free to exceed 1, and because A_hat
          has eigenvalues in [-1,1] (negative ones too), the Kronecker product
          W (x) A_hat can have rho >= 1 even though sym(I-J_z) >= m I.
          ==> genuinely different contraction certificate from the IGNN.

DIFFERENCE 2 -- the solver (so the FORWARD genuinely differs from Picard):
  IGNN:   plain Picard  z <- sigma(A_hat z W^T + Xu).
  MonDEQ: forward-backward OPERATOR SPLITTING (the canonical MonDEQ solver,
          Winston-Kolter 2020).  With operator split  T(z) = sigma(W z + b)
          written as the monotone-operator problem  0 in (F + G)(z),
          the FB / proximal-gradient iteration with step alpha is
            z_{k+1} = prox_{alpha G}( z_k - alpha (z_k - W z_k - b) )
                    = relu( (1-alpha) z_k + alpha (W z_k + b) ),
          i.e. a DAMPED (averaged) fixed-point iteration with relaxation alpha
          in (0,1].  At alpha=1 it reduces to Picard; we use alpha<1 (under-
          relaxation), the standard MonDEQ choice that converges for monotone
          operators where Picard need not.  ReLU is the proximal operator of
          the indicator of the nonneg orthant, so prox_{alpha G}=relu exactly.
          ==> genuinely different solver from the IGNN's plain Picard.

HONEST SCOPE NOTE (stated, per task): an EXACT all-of-(W(x)A_hat) monotone
parameterization would require coupling W to the spectrum of A_hat (incl. its
NEGATIVE eigenvalues); that is impractical to keep differentiable and is exactly
the source of the rho>=1 concern.  We therefore use the standard WK parameter-
ization on the CHANNEL matrix (the principled, published MonDEQ choice) and then
MEASURE NUMERICALLY whether (a) I-J_z is actually m-strongly monotone
(min eig of sym(I-J_z) >= m) and (b) rho(J_z) < 1.  Discovering that (a) can
hold while (b) FAILS is the scientific content of the probe.

WHAT WE MEASURE / TEST (tasks 2-4)
----------------------------------
  T2. rho(J_z) (decisive), ||J_z||_2, and monotonicity margin
      m = lambda_min(sym(I - J_z)).  All three reported; verdict keys on rho.
  T3. Run AEGIS S_c matrix-free (ScalableSensitivity, UNMODIFIED, operator-
      agnostic as in paper/review/universal_findings.md): does the truncated
      Neumann converge?  Then VALIDATE sigma_1 against a DENSE ground-truth
      SVD of the explicitly-formed S_c on a SMALL MonDEQ (methodology of
      scripts/_probe_aegis_sigma1.py).  Report relative error.
  T4. If S_c works: sanity-check that the SVD-optimal v_1 edge direction moves
      z* more than a random edge direction (diagnostics are meaningful).

VERDICT
-------
  FEASIBLE   iff rho(J_z) < 1 on the trained model AND matrix-free sigma_1
             matches the dense ground truth (< ~1%).
  INFEASIBLE if rho(J_z) >= 1 (Neumann diverges).  This is a VALID, important
             negative finding: AEGIS's contraction assumption is ESSENTIAL;
             monotonicity alone is INSUFFICIENT.  We do NOT force a workaround
             that abandons the matrix-free machinery.

Run:  .venv/bin/python scripts/exp_mondeq_probe.py            (full: train Cora)
      .venv/bin/python scripts/exp_mondeq_probe.py --smoke    (tiny, CPU, fast)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_func

ROOT = Path("/home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN")
sys.path.insert(0, str(ROOT))

# Reuse the EXACT Cora loader from the IGNN example so data/normalization match.
from iem.examples.ignn_cora import _download_cora, _load_cora  # noqa: E402
# AEGIS matrix-free operator -- UNMODIFIED, exactly as the IGNN uses it.
from iem.scalable import ScalableSensitivity  # noqa: E402


# ======================================================================
# Monotone graph DEQ (MonDEQ-style).  See module docstring for the math.
# ======================================================================
class GraphMonDEQ(nn.Module):
    r"""Z* = sigma(A_hat Z* W^T + X U), W = Winston-Kolter monotone param,
    solved by forward-backward (averaged) operator splitting with relaxation
    alpha in (0,1].  Exposes the SAME operator interface as the IGNN
    (``operator(Z, ctx)`` with ctx={'A_hat','X_proj'}; ``forward`` returns
    (logits, Z_star, ctx)) so ScalableSensitivity differentiates it unchanged.

    NOTE: ScalableSensitivity builds J_z/J_A from ``operator`` (the callable it
    differentiates), which is the PLAIN fixed-point map sigma(A_hat Z W^T+Xu).
    The FB relaxation alpha is a SOLVER detail used only to FIND z*; the
    equilibrium z* and its Jacobians are identical to the un-relaxed map's
    (the averaged map sigma_alpha and sigma share the same fixed point and, at
    that fixed point, J_z(averaged) = (1-alpha)I + alpha J_z so they share
    spectral structure shifted/scaled -- but AEGIS differentiates the un-
    relaxed ``operator``, the physically meaningful equilibrium map, exactly as
    it does for the IGNN).  rho(J_z) below is therefore the spectral radius of
    the SAME object AEGIS's Neumann series must invert.
    """

    def __init__(self, n_features: int, hidden: int, n_classes: int,
                 m: float = 0.05, alpha: float = 0.5, dropout: float = 0.5,
                 skew_scale: float = 1.0):
        super().__init__()
        self.hidden = hidden
        self.m = m                # strong-monotonicity margin target (channel)
        self.alpha = alpha        # FB relaxation in (0,1]; <1 => not Picard
        self.dropout = dropout
        self.skew_scale = skew_scale  # multiplies the skew block; non-normality knob
        self.U = nn.Linear(n_features, hidden)        # input projection X U
        # WK monotone parameter blocks (channel space, hidden x hidden):
        #   W = (1-m) I - A_par^T A_par + s*(B_par - B_par^T)
        self.A_par = nn.Parameter(torch.empty(hidden, hidden))
        self.B_par = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_normal_(self.A_par, gain=0.5)
        nn.init.xavier_normal_(self.B_par, gain=0.5)
        self.head = nn.Linear(hidden, n_classes)

    def W_eff(self) -> torch.Tensor:
        """Winston-Kolter monotone channel matrix.
        sym(W) = (1-m)I - A^T A  <=  (1-m) I  =>  I - W is m-strongly monotone
        IN CHANNEL SPACE.  The skew part s*(B-B^T) leaves sym(W) untouched but
        makes W (hence W (x) A_hat) NON-NORMAL: this is exactly the knob by which
        rho(J_z) can decouple from -- and exceed -- the monotone margin/Re(lambda)
        bound (the probe's central concern).  s=skew_scale.
        """
        I = torch.eye(self.hidden, device=self.A_par.device, dtype=self.A_par.dtype)
        ATA = self.A_par.t() @ self.A_par                 # PSD
        skew = self.B_par - self.B_par.t()                # skew-symmetric
        return (1.0 - self.m) * I - ATA + self.skew_scale * skew

    def operator(self, Z: torch.Tensor, ctx: dict) -> torch.Tensor:
        """F(Z) = ReLU(A_hat @ Z @ W_eff^T + X_proj).  The un-relaxed
        equilibrium map AEGIS differentiates (Jacobian source)."""
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return F_func.relu(A_hat @ (Z @ self.W_eff().t()) + X_proj)

    def operator_fb(self, Z: torch.Tensor, ctx: dict) -> torch.Tensor:
        """One forward-backward (averaged) step with relaxation alpha:
        Z <- relu( (1-alpha) Z + alpha (A_hat Z W^T + X_proj) ).
        prox of the nonneg-orthant indicator is relu, so this is the exact
        proximal-gradient / FB step for the monotone-operator equilibrium."""
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        lin = A_hat @ (Z @ self.W_eff().t()) + X_proj
        return F_func.relu((1.0 - self.alpha) * Z + self.alpha * lin)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor,
                max_iter: int = 300, tol: float = 1e-6, train_dropout: bool = False):
        N = X.shape[0]
        X_proj = self.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}
        # FORWARD-BACKWARD operator splitting (NOT plain Picard).
        Z = torch.zeros(N, self.hidden, device=X.device, dtype=X_proj.dtype)
        for _ in range(max_iter):
            Z_new = self.operator_fb(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm(), 1.0):
                Z = Z_new
                break
            Z = Z_new
        Z_star = Z
        H = Z_star
        if train_dropout and self.dropout > 0:
            H = F_func.dropout(H, p=self.dropout, training=True)
        logits = self.head(H)
        return logits, Z_star, ctx


# ======================================================================
# Helpers: active edge basis (identical to ScalableSensitivity / aegis_sigma1)
# ======================================================================
def active_edge_index(A: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    iu = torch.triu_indices(N, N, offset=1, device=A.device)
    active = A[iu[0], iu[1]].abs() > 1e-10
    return iu[:, active].t().contiguous().to(torch.long)


# ======================================================================
# DENSE ground-truth S_c (methodology of scripts/_probe_aegis_sigma1.py)
# Forms J_z, J_A explicitly, builds S_c = (I - J_z)^{-1} J_A P_c, SVD.
# The edge basis (P_c: both (i,j) and (j,i)) MATCHES ScalableSensitivity.
# ======================================================================
def _rho_power_dense(M: torch.Tensor, iters: int = 500) -> float:
    """Spectral radius of a dense matrix via power iteration + Rayleigh quotient
    (LAPACK-free fallback for matrices where eigvals balks)."""
    torch.manual_seed(0)
    v = torch.randn(M.shape[0], dtype=M.dtype)
    v = v / v.norm()
    for _ in range(iters):
        w = M @ v
        nw = w.norm()
        if nw < 1e-14:
            return 0.0
        v = w / nw
    return abs(float((v @ (M @ v))))


def dense_Sc_groundtruth(model: GraphMonDEQ, X: torch.Tensor, A: torch.Tensor,
                         zstar: torch.Tensor, op_name: str = "operator_fb"):
    """Dense ground-truth S_c for the operator ``op_name``.

    AEGIS must differentiate the SAME map whose fixed point z* is (else J_z is
    taken at a non-equilibrium and S_c is meaningless).  The MonDEQ's ACTUAL
    operator is the forward-backward averaged map ``operator_fb`` (the forward
    solves it; ReLU is nonlinear so its fixed point differs from the plain
    ``operator``'s).  We therefore default to linearizing ``operator_fb``; the
    decisive rho(J_z) below is the spectral radius of THAT operator's Jacobian,
    which is exactly what AEGIS's Neumann series must invert.
    """
    N, hid = zstar.shape
    ctx = {"A_hat": A, "X_proj": model.U(X)}
    op = getattr(model, op_name)

    def F_of_z(zf):
        return op(zf.reshape(N, hid), ctx).reshape(-1)

    def F_of_A(Av):
        return op(zstar, {"A_hat": Av, "X_proj": model.U(X)}).reshape(-1)

    Nh = N * hid
    Jz = torch.autograd.functional.jacobian(F_of_z, zstar.reshape(-1))      # (Nh,Nh)
    JA = torch.autograd.functional.jacobian(F_of_A, A)                      # (Nh,N,N)

    edge_idx = active_edge_index(A)
    E = edge_idx.shape[0]
    JAPc = torch.zeros(Nh, E, dtype=Jz.dtype)
    for k in range(E):
        i, j = edge_idx[k].tolist()
        JAPc[:, k] = JA[:, i, j] + JA[:, j, i]                              # P_c (both)

    Ieye = torch.eye(Nh, dtype=Jz.dtype)
    if not torch.isfinite(Jz).all():
        # FB forward diverged (rho>=1 region with too-large step): z* blew up.
        # Report rho via the (finite) power iteration on J_z's action instead;
        # but if J_z itself is non-finite the model is unusable -> flag.
        return dict(rho=float("inf"), opnorm=float("inf"), mono_m=float("nan"),
                    E=E, Nh=Nh, neumann_divergent=True, sigma1_dense=float("nan"),
                    edge_idx=edge_idx, nonfinite_Jz=True)
    # rho via eig; fall back to power iteration if LAPACK balks on a hard matrix.
    try:
        eig = torch.linalg.eigvals(Jz)
        rho = float(eig.abs().max())
    except Exception:
        rho = _rho_power_dense(Jz)
    try:
        opn = float(torch.linalg.svdvals(Jz)[0])
    except Exception:
        opn = _rho_power_dense(Jz.t() @ Jz) ** 0.5
    sym = 0.5 * ((Ieye - Jz) + (Ieye - Jz).t())                            # sym(I-Jz)
    try:
        mono_m = float(torch.linalg.eigvalsh(sym).min())                    # min eig
    except Exception:
        # -rho(-sym shifted): min eig of sym = -max eig of (cI - sym) + c
        c = float(sym.diag().abs().sum()) + 1.0
        mono_m = c - _rho_power_dense(c * Ieye - sym)

    info = dict(rho=rho, opnorm=opn, mono_m=mono_m, E=E, Nh=Nh)
    if rho >= 1.0:
        # (I - Jz) may still be invertible (no eigenvalue exactly 1), so the
        # DENSE S_c can be formed by exact solve -- but the NEUMANN series that
        # AEGIS uses will DIVERGE.  Form the dense S_c anyway (ground truth)
        # iff (I - Jz) is non-singular; flag the Neumann incompatibility.
        info["neumann_divergent"] = True
    else:
        info["neumann_divergent"] = False

    # Exact dense S_c via direct solve (valid whenever I-Jz is non-singular).
    try:
        Sc = torch.linalg.solve(Ieye - Jz, JAPc)
        svals = torch.linalg.svdvals(Sc)
        info["sigma1_dense"] = float(svals[0])
        info["sigma2_dense"] = float(svals[1]) if E > 1 else float("nan")
        info["Sc"] = Sc
        info["edge_idx"] = edge_idx
    except Exception as e:  # singular (I - Jz)
        info["sigma1_dense"] = float("nan")
        info["solve_error"] = str(e)
        info["edge_idx"] = edge_idx
    return info


# ======================================================================
# Matrix-free sigma_1 via the UNMODIFIED AEGIS ScalableSensitivity operator.
# Operator-agnostic: we hand it the MonDEQ's `operator` as F, exactly as the
# RL Bellman fixed point did in paper/review/universal_findings.md.
# ======================================================================
def matrixfree_sigma1(model: GraphMonDEQ, X: torch.Tensor, A: torch.Tensor,
                      zstar: torch.Tensor, neumann_terms: int,
                      svd_k: int = 6, n_power_iter: int = 30,
                      op_name: str = "operator_fb"):
    op_fn = getattr(model, op_name)

    def F_op(z, c):
        return op_fn(z, c)

    ctx = {"A_hat": A.detach(), "X_proj": model.U(X).detach()}
    op = ScalableSensitivity(F_op, zstar.detach(), ctx,
                             A_key="A_hat", neumann_terms=neumann_terms)
    k = min(svd_k, op.num_edges)
    U, sigma, Vh = op.top_k_svd(k=k, n_power_iter=n_power_iter)
    return op, float(sigma[0]), Vh[0].detach()


# ======================================================================
# T4 sanity: does the SVD-optimal edge direction v_1 move z* more than a
# random edge direction (matched edge-norm)?  Reconverge with FB solver.
# ======================================================================
def reconverge(model: GraphMonDEQ, X: torch.Tensor, A_pert: torch.Tensor,
               max_iter: int = 600, tol: float = 1e-7) -> torch.Tensor:
    with torch.no_grad():
        _, Z_star, _ = model(X, A_pert, max_iter=max_iter, tol=tol)
    return Z_star


def edge_vec_to_dA(v: torch.Tensor, edge_idx: torch.Tensor, N: int) -> torch.Tensor:
    dA = torch.zeros(N, N, device=v.device, dtype=v.dtype)
    dA[edge_idx[:, 0], edge_idx[:, 1]] = v
    dA[edge_idx[:, 1], edge_idx[:, 0]] = v
    return dA


def v1_vs_random(model, X, A, zstar, v1, edge_idx, eps=0.1, seed=0):
    N = A.shape[0]
    v1u = v1 / (v1.norm() + 1e-30)
    dA_svd = edge_vec_to_dA(eps * v1u, edge_idx, N)
    z_svd = reconverge(model, X, A + dA_svd)
    d_svd = float((z_svd - zstar).norm())

    g = torch.Generator(device=v1.device).manual_seed(seed)
    rnd = torch.randn(v1.shape, generator=g, device=v1.device, dtype=v1.dtype)
    rnd = rnd / (rnd.norm() + 1e-30)
    dA_rnd = edge_vec_to_dA(eps * rnd, edge_idx, N)
    z_rnd = reconverge(model, X, A + dA_rnd)
    d_rnd = float((z_rnd - zstar).norm())
    return d_svd, d_rnd


# ======================================================================
# Training
# ======================================================================
def train_mondeq(model, X, A, y, train_mask, val_mask, test_mask,
                 epochs=200, lr=0.01, wd=5e-4, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A, train_dropout=True)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A)
                pred = logits.argmax(1)
                va = float((pred[val_mask] == y[val_mask]).float().mean())
                ta = float((pred[test_mask] == y[test_mask]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep {ep:3d}  loss {float(loss):.4f}  val {va:.3f}  test {ta:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ======================================================================
# SMOKE: tiny random graph MonDEQ, CPU.  Validates the whole pipeline
# (monotone param, FB solver, dense S_c == matrix-free sigma_1) fast.
# ======================================================================
def run_smoke():
    print("=== SMOKE: tiny MonDEQ, dense-vs-matrix-free sigma_1 ===", flush=True)
    torch.manual_seed(0)
    N, nf, hid, nc = 10, 5, 4, 3
    # sparse symmetric A_hat with exact zeros (well-defined active-edge set)
    M = (torch.rand(N, N) < 0.35).float()
    M = torch.triu(M, 1); M = M + M.t()
    deg = M.sum(1) + 1.0
    dis = deg.pow(-0.5)
    A = (dis.unsqueeze(1) * (M + torch.eye(N)) * dis.unsqueeze(0))
    A = A.float()
    X = torch.randn(N, nf)
    model = GraphMonDEQ(nf, hid, nc, m=0.05, alpha=0.5).double()
    X = X.double(); A = A.double()
    with torch.no_grad():
        _, zstar, _ = model(X, A, max_iter=2000, tol=1e-12)
    # residual of the FB operator (the map AEGIS differentiates) -- self-consistent.
    ctxs = {"A_hat": A, "X_proj": model.U(X)}
    res = float((model.operator_fb(zstar, ctxs) - zstar).norm())
    print(f"  N={N} hid={hid}  FB fixed-point residual = {res:.2e}", flush=True)

    gt = dense_Sc_groundtruth(model, X, A, zstar)
    print(f"  rho(J_z)={gt['rho']:.4f}  ||J_z||2={gt['opnorm']:.4f}  "
          f"mono_m=lambda_min(sym(I-Jz))={gt['mono_m']:.4f}  E={gt['E']}", flush=True)
    print(f"  Neumann divergent (rho>=1)? {gt['neumann_divergent']}", flush=True)

    # matrix-free: deep Neumann; auto K if rho<1 else big cap
    nt = 0 if gt["rho"] < 1.0 else 3000
    _, sig_mf, _ = matrixfree_sigma1(model, X, A, zstar,
                                     neumann_terms=(nt if nt else 200),
                                     n_power_iter=40)
    sig_gt = gt["sigma1_dense"]
    rel = abs(sig_gt - sig_mf) / (abs(sig_gt) + 1e-30) * 100.0
    print(f"  sigma1 DENSE (GT) = {sig_gt:.6f}   sigma1 matrix-free = {sig_mf:.6f}", flush=True)
    print(f"  relative error = {rel:.4f}%   "
          f"VERDICT: {'PASS' if rel < 1.0 else 'FAIL'} (smoke machinery)", flush=True)
    return rel < 1.0


# ======================================================================
# FULL: train MonDEQ on Cora, measure rho/||Jz||/m, run + validate AEGIS S_c.
# Full Cora N=2708 -> Nh=2708*hid is too large to form J_z densely; for the
# DENSE ground-truth validation we use a BFS ego-subgraph (same tactic the IGNN
# example uses for its dense Jacobian).  rho/||Jz||/m on the FULL graph come
# from matrix-free power iteration; the dense<->matrix-free sigma_1 agreement is
# validated on the tractable subgraph.  Reported separately and labeled.
# ======================================================================
def run_full(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== FULL probe on {device} ===", flush=True)
    data_dir = ROOT / "datasets/cora"
    _download_cora(data_dir)
    data = _load_cora(data_dir)
    X = data["X"].to(device); A = data["A_hat"].to(device); y = data["y"].to(device)
    tr, va, te = data["train_mask"].to(device), data["val_mask"].to(device), data["test_mask"].to(device)
    print(f"  Cora N={data['N']} feat={data['n_features']} classes={data['n_classes']}", flush=True)

    torch.manual_seed(args.seed)
    model = GraphMonDEQ(data["n_features"], args.hidden, data["n_classes"],
                        m=args.m, alpha=args.alpha, dropout=0.5,
                        skew_scale=args.skew).to(device)
    print(f"  GraphMonDEQ hidden={args.hidden} m={args.m} alpha={args.alpha} "
          f"skew_scale={args.skew} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("--- train (forward-backward operator splitting solver) ---", flush=True)
    model = train_mondeq(model, X, A, y, tr, va, te, epochs=args.epochs)
    model.eval()
    with torch.no_grad():
        # Generous forward so the full-graph FB equilibrium is genuinely reached
        # (rho measured off-equilibrium is meaningless).  alpha<0.5 needed:
        # alpha=0.5 oscillates on full Cora (residual plateaus ~0.34).
        logits, Z_star_full, ctx = model(X, A, max_iter=5000, tol=1e-7)
        pred = logits.argmax(1)
        test_acc = float((pred[te] == y[te]).float().mean())
        res_fb = float((model.operator_fb(Z_star_full, ctx) - Z_star_full).norm())
        res_plain = float((model.operator(Z_star_full, ctx) - Z_star_full).norm())
    converged = res_fb < 1e-3 * max(float(Z_star_full.norm()), 1.0)
    print(f"  final test acc = {test_acc:.3f}   FB-operator residual = {res_fb:.2e} "
          f"(AEGIS differentiates THIS; converged? {converged});  "
          f"plain-operator residual at FB-fp = {res_plain:.2e}", flush=True)
    if not converged:
        print("  !! WARNING: full-graph equilibrium NOT reached -> rho measured "
              "off-equilibrium; lower --alpha (e.g. 0.3) for convergence.", flush=True)

    # ---- T2 on the FULL graph: rho / ||Jz|| via matrix-free power iteration ----
    # DECISIVE rho is for operator_fb (the map AEGIS's Neumann inverts).  We ALSO
    # report rho of the plain operator for context (the 'would-Picard-converge'
    # number); FB averaging with alpha<1 shrinks the spectrum toward contraction.
    def F_fb(z, c):
        return model.operator_fb(z, c)

    def F_plain(z, c):
        return model.operator(z, c)

    op_fb = ScalableSensitivity(F_fb, Z_star_full.detach(),
                                {"A_hat": A.detach(), "X_proj": ctx["X_proj"].detach()},
                                A_key="A_hat", neumann_terms=1)
    op_plain = ScalableSensitivity(F_plain, Z_star_full.detach(),
                                   {"A_hat": A.detach(), "X_proj": ctx["X_proj"].detach()},
                                   A_key="A_hat", neumann_terms=1)
    rho_full = rho_rayleigh(op_fb, iters=300)          # DECISIVE (FB operator)
    opn_full = opnorm_power(op_fb, iters=300)
    rho_plain = rho_rayleigh(op_plain, iters=300)      # context (plain operator)
    opn_plain = opnorm_power(op_plain, iters=300)
    print("\n[T2] FULL-GRAPH spectral measurements (matrix-free power iteration)", flush=True)
    print(f"     rho(J_z^FB)    = {rho_full:.5f}   (DECISIVE: AEGIS Neumann converges iff <1)", flush=True)
    print(f"     ||J_z^FB||_2   = {opn_full:.5f}", flush=True)
    print(f"     rho(J_z^plain) = {rho_plain:.5f}   ||J_z^plain||_2 = {opn_plain:.5f}  (context)", flush=True)
    print(f"     rho(J_z^FB) < 1 ?  {rho_full < 1.0}", flush=True)

    # ---- DENSE ground-truth validation on a BFS ego-subgraph (tractable) ----
    # ALL subgraph dense/matrix-free work is done on CPU+float64 (clean inverse,
    # no GPU contention).  Move the whole model to cpu/double for this block.
    from iem.adversarial import extract_ego_subgraph
    sub = extract_ego_subgraph(A, max_nodes=args.sub_nodes).to(device)
    A_sub = A[sub][:, sub].contiguous().double().cpu()
    X_sub = X[sub].contiguous().double().cpu()
    md = model.to("cpu").double()
    with torch.no_grad():
        _, z_sub, _ = md(X_sub, A_sub, max_iter=2000, tol=1e-10)
    gt = dense_Sc_groundtruth(md, X_sub, A_sub, z_sub)
    print(f"\n[T2-sub] subgraph S={sub.shape[0]} E={gt['E']}  "
          f"rho(J_z)={gt['rho']:.5f}  ||J_z||2={gt['opnorm']:.5f}  "
          f"mono_m=lambda_min(sym(I-Jz))={gt['mono_m']:.5f}", flush=True)
    print(f"         monotone (m>0)? {gt['mono_m'] > 0}   "
          f"Neumann divergent (rho>=1)? {gt['neumann_divergent']}", flush=True)

    # ---- T3 matrix-free sigma_1 on the SAME subgraph, compare to dense GT ----
    nt = 3000 if gt["rho"] >= 0.98 else 0
    _, sig_mf, v1 = matrixfree_sigma1(md, X_sub, A_sub, z_sub,
                                      neumann_terms=(nt if nt else 200),
                                      n_power_iter=40)
    sig_gt = gt["sigma1_dense"]
    rel = abs(sig_gt - sig_mf) / (abs(sig_gt) + 1e-30) * 100.0 if sig_gt == sig_gt else float("nan")
    print("\n[T3] AEGIS S_c matrix-free vs DENSE ground truth (subgraph)", flush=True)
    print(f"     sigma1 DENSE (GT)     = {sig_gt:.6f}", flush=True)
    print(f"     sigma1 matrix-free    = {sig_mf:.6f}", flush=True)
    print(f"     relative error        = {rel:.4f}%", flush=True)

    # ---- T4 sanity: v1 vs random edge direction (subgraph, only if S_c ok) ----
    d_svd = d_rnd = float("nan")
    if gt["rho"] < 1.0 and rel == rel and rel < 1.0:
        d_svd, d_rnd = v1_vs_random(md, X_sub, A_sub, z_sub, v1, gt["edge_idx"],
                                    eps=args.eps)
        print("\n[T4] SVD-optimal v1 vs random edge direction (reconverged |dz*|)", flush=True)
        print(f"     |dz*| v1 = {d_svd:.5f}   |dz*| random = {d_rnd:.5f}   "
              f"ratio = {d_svd / (d_rnd + 1e-30):.2f}x", flush=True)

    # ---- VERDICT ----
    rho_decisive = rho_full
    feasible = (rho_decisive < 1.0) and (rel == rel) and (rel < 1.0)
    print("\n" + "=" * 64, flush=True)
    print("VERDICT", flush=True)
    print(f"  rho(J_z) full graph     = {rho_decisive:.5f}  ({'<1 OK' if rho_decisive<1 else '>=1 DIVERGES'})", flush=True)
    print(f"  matrix-free sigma1 err  = {rel:.4f}%", flush=True)
    print(f"  ==> {'FEASIBLE' if feasible else 'INFEASIBLE'}", flush=True)
    print("=" * 64, flush=True)

    return dict(test_acc=test_acc, rho_full=rho_full, opn_full=opn_full,
                rho_sub=gt["rho"], opn_sub=gt["opnorm"], mono_m=gt["mono_m"],
                sigma1_dense=sig_gt, sigma1_mf=sig_mf, rel_err=rel,
                d_svd=d_svd, d_rnd=d_rnd, feasible=feasible,
                neumann_divergent=gt["neumann_divergent"])


# --- spectral-radius / operator-norm via matrix-free power iteration on J_z ---
def rho_rayleigh(op: ScalableSensitivity, iters: int = 300) -> float:
    """rho(J_z) via power iteration + (sign-aware) Rayleigh quotient.
    Identical method to scripts/exp_fullgraph_attack_table.rho_rayleigh."""
    torch.manual_seed(0)
    v = torch.randn(op.D, device=op.device, dtype=op.dtype)
    v = v / v.norm()
    for _ in range(iters):
        Jv = op._jvp_Jz(v)
        nv = Jv.norm()
        if nv < 1e-12:
            return 0.0
        v = Jv / nv
    return abs(float((v * op._jvp_Jz(v)).sum().item()))


def opnorm_power(op: ScalableSensitivity, iters: int = 300) -> float:
    """||J_z||_2 via power iteration on J_z^T J_z."""
    torch.manual_seed(1)
    v = torch.randn(op.D, device=op.device, dtype=op.dtype)
    v = v / v.norm()
    s = 0.0
    for _ in range(iters):
        Jv = op._jvp_Jz(v)
        w = op._vjp_Jz(Jv)
        s = float(w.norm().sqrt() if False else (Jv.norm()))
        nv = w.norm()
        if nv < 1e-12:
            break
        v = w / nv
    # final Rayleigh for the singular value
    Jv = op._jvp_Jz(v)
    return float(Jv.norm())


# ======================================================================
# GRID: train several principled monotone configs (vary m, alpha, skew_scale)
# and tabulate (test_acc, mono_m, rho, ||Jz||2) on the subgraph linearization.
# Answers: can a TRAINED, accurate, genuinely-monotone (mono_m>0) graph DEQ
# reach rho(J_z) >= 1 ?  The skew_scale knob exercises the non-normality that
# (per the linear analysis) decouples rho from the monotone margin.
# ======================================================================
def measure_on_subgraph(model, X, A, sub_nodes, device):
    from iem.adversarial import extract_ego_subgraph
    sub = extract_ego_subgraph(A, max_nodes=sub_nodes).to(device)
    A_sub = A[sub][:, sub].contiguous().double().cpu()
    X_sub = X[sub].contiguous().double().cpu()
    # Dense GT (inverse/eig) on CPU+float64: move the WHOLE model to cpu/double.
    md = model.to("cpu").double()
    with torch.no_grad():
        _, z_sub, _ = md(X_sub, A_sub, max_iter=2000, tol=1e-10)
    gt = dense_Sc_groundtruth(md, X_sub, A_sub, z_sub)
    model.to(device).float()
    return gt, sub.shape[0]


def run_grid(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== GRID probe on {device} ===", flush=True)
    data = _load_cora(ROOT / "datasets/cora")
    X = data["X"].to(device); A = data["A_hat"].to(device); y = data["y"].to(device)
    tr, va, te = (data["train_mask"].to(device), data["val_mask"].to(device),
                  data["test_mask"].to(device))
    # alpha=0.3 (converging) baseline; vary skew (non-normality) and m.  We probe
    # whether a CONVERGED, accurate, genuinely-monotone (mono_m>0) model can hit
    # rho(J_z^FB) >= 1.  Reported on the subgraph FB linearization at equilibrium.
    configs = [
        dict(m=0.05, alpha=0.3, skew=1.0),
        dict(m=0.05, alpha=0.3, skew=2.0),
        dict(m=0.05, alpha=0.3, skew=4.0),
        dict(m=0.02, alpha=0.3, skew=2.0),
        dict(m=0.10, alpha=0.3, skew=3.0),
        dict(m=0.05, alpha=0.5, skew=1.0),   # alpha=0.5 (may not fully converge)
    ]
    print(f"{'m':>5}{'alpha':>6}{'skew':>5}{'test_acc':>9}{'mono_m':>9}"
          f"{'rho(Jz)':>9}{'||Jz||2':>9}{'mono>0':>7}{'rho>=1':>7}", flush=True)
    rows = []
    any_mono_and_rho_ge1 = False
    for cfg in configs:
        try:
            torch.manual_seed(args.seed)
            model = GraphMonDEQ(data["n_features"], args.hidden, data["n_classes"],
                                m=cfg["m"], alpha=cfg["alpha"], dropout=0.5,
                                skew_scale=cfg["skew"]).to(device)
            model = train_mondeq(model, X, A, y, tr, va, te, epochs=args.epochs,
                                 verbose=False)
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A, max_iter=5000, tol=1e-7)
                ta = float((logits.argmax(1)[te] == y[te]).float().mean())
            gt, S = measure_on_subgraph(model, X, A, args.sub_nodes, device)
        except Exception as e:
            print(f"{cfg['m']:>5}{cfg['alpha']:>6}{cfg['skew']:>5}   ERROR: {type(e).__name__}: {str(e)[:60]}",
                  flush=True)
            continue
        mono_ok = gt["mono_m"] > 0
        rho_ge1 = gt["rho"] >= 1.0
        any_mono_and_rho_ge1 = any_mono_and_rho_ge1 or (mono_ok and rho_ge1)
        print(f"{cfg['m']:>5}{cfg['alpha']:>6}{cfg['skew']:>5}{ta:>9.3f}"
              f"{gt['mono_m']:>9.4f}{gt['rho']:>9.4f}{gt['opnorm']:>9.4f}"
              f"{str(mono_ok):>7}{str(rho_ge1):>7}", flush=True)
        rows.append(dict(**cfg, test_acc=ta, mono_m=gt["mono_m"], rho=gt["rho"],
                         opn=gt["opnorm"], S=S))
    print("\nGRID SUMMARY", flush=True)
    print(f"  any trained model with (mono_m>0 AND rho>=1)? {any_mono_and_rho_ge1}", flush=True)
    print(f"  any trained model with rho>=1 (Neumann diverges)? "
          f"{any(r['rho']>=1.0 for r in rows)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU validation only")
    ap.add_argument("--grid", action="store_true", help="train a config grid, tabulate rho/mono")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--m", type=float, default=0.05, help="monotonicity margin target")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="FB relaxation in (0,1]; <1 => not Picard. 0.3 converges "
                         "on full Cora (0.5 oscillates -> non-convergence).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-nodes", type=int, default=40, help="ego-subgraph for dense GT")
    ap.add_argument("--eps", type=float, default=0.1, help="T4 perturbation budget")
    ap.add_argument("--skew", type=float, default=2.0, help="skew_scale (non-normality knob)")
    args = ap.parse_args()

    if args.smoke:
        ok = run_smoke()
        sys.exit(0 if ok else 1)
    if args.grid:
        run_grid(args)
        return
    run_full(args)


if __name__ == "__main__":
    main()
