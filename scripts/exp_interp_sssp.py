#!/usr/bin/env python
"""DECISIVE INTERPRETABILITY GATE — does the equilibrium resolvent (I - J_z)^{-1}
/ constrained sensitivity S_c MECHANISTICALLY DECODE what an implicit GNN computes,
on SINGLE-SOURCE SHORTEST-PATH (SSSP)?

WHY SSSP (and not connected-components). The CC gate (scripts/exp_interp_smoke.py)
was WEAK precisely because its causal structure == adjacency support: "which edges
reach u" is, by definition, the connected edges, so EVERY method (resolvent,
input-gradient, raw adjacency) trivially recovers it and they all tie. SSSP is the
opposite: the answer is the shortest-path TREE — a strict WEIGHTED subset of edges,
NOT reducible to adjacency support or graph spectrum. An edge incident to u that is
NOT on u's shortest path should have SMALL influence on d(s,u); the tree edges on the
s->u path should DOMINATE. So SSSP is a clean test of whether the resolvent decodes
*structure the adjacency does not already give away*.

CONSTRUCTION (the crux: make dz*/dw_e a literal S_c column).
We use an UNNORMALIZED weighted propagation operator
    z* = phi(A_w z* W^T + U(X)),  phi = relu,  A_w[i,j] = w_ij  (raw edge weight).
Then for edge e=(i,j), partial z* / partial w_e is EXACTLY the constrained
sensitivity column S_c[:,e] = (I-J_z)^{-1}(J_A[:,iN+j] + J_A[:,jN+i]) produced by the
bug-audited iem.adversarial machinery -- no GCN renormalization smears one physical
weight across many A_hat entries (which would muddy "edge-weight sensitivity"). kappa
is enforced on the product ||W||*||A_w|| exactly as in the phase/CC experiments, just
against A_w's spectral norm instead of A_hat's.

REUSED, BUG-AUDITED MACHINERY (verified bit-identical / bit-exact in prior work):
  - IGNN_Kappa, set_seed                          (exp_phase_transition)
  - jz_kron = diag(vec mask) kron(A_w, W)         (exp_reachability; == compute_jacobian)
  - _compute_structural_jacobian / S / S_c        (iem.adversarial; the C3 path of the
                                                   CC gate, validated cos/rel-err there)
  - resolvent block vs autograd dz*/d(.)          (S2 self-check, rel-err 4e-16 in CC)

GATES (seed 42), reported honestly:
  G1   expressivity-under-contraction: does a kappa<1 IGNN learn SSSP? rho(J_z),
       distance correlation / MAE on held-out nodes & graphs. SSSP's min-plus fixed
       point is near-marginal; if kappa<1 CANNOT fit it, that is the
       contraction<->expressivity wall (a real finding -> report loudly).
  C1'  THE discriminating test -- SP-TREE recovery. For each node u rank edges by
       resolvent gain ||dz*_u/dw_e||. Do top edges recover u's shortest-PATH edge set
       (edges actually on the s->u path), NOT merely all edges incident to u? Report
       AUC/precision vs baselines: (i) adjacency-incidence, (ii) input-gradient
       ||d y_hat_u / d w_e||, (iii) graph-distance heuristic. Pivot is REAL only if the
       resolvent BEATS these (CC tied them; SSSP should separate IF decoding is real).
  C3'  causal tree-vs-nontree: perturb a TREE edge weight vs a NON-TREE edge weight;
       does the resolvent predict large distance change for tree edges, ~zero for
       non-tree, MATCHING the re-solved nonlinear equilibrium? Report separation.
  C2'  first look: do S_c / resolvent eigen-modes reveal path-tree organization beyond
       the adjacency spectrum (alignment vs an adjacency-only control)?

SELF-CHECKS:
  S1  Dijkstra (networkx) vs scipy.csgraph.dijkstra agree exactly on distances + tree.
  S2  resolvent block (I-J_z)^{-1}(J_x col) matches autograd dz*/d(input) (reused path).
  S3  S_c column (I-J_z)^{-1}(J_A) matches autograd dz*/dw_e for one edge (the C1' path).

OUTPUT:
  scripts/exp_interp_sssp.py, results/exp_interp_sssp.csv, paper/review/interp_sssp_findings.md

Usage:
    ./.venv/bin/python scripts/exp_interp_sssp.py
    (venv is NOT inherited by detached shells -- always call ./.venv/bin/python)
"""
from __future__ import annotations

import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iem.adversarial import (  # noqa: E402
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    structural_sensitivity_matrix,
)
from scripts.exp_phase_transition import IGNN_Kappa, set_seed  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

SEED = 42
DTYPE = torch.float64  # double precision for clean Jacobian / resolvent linear algebra
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Weighted graph generation: connected ER-style graph with RANDOM log-uniform edge
# weights INDEPENDENT of topology. This is the crux of the non-definitional design.
#
# A random GEOMETRIC graph was tried first and REJECTED: there the weighted
# shortest-path distance is ~collinear with hop-count distance (corr ~0.92), so SSSP
# degenerates back into BFS/reachability -- the exact definitional trap that made the
# CC gate weak, just disguised. With log-uniform weights decoupled from topology,
# corr(weighted-dist, hop-dist) drops to ~0.57 (paths genuinely REORDER), and the
# shortest-path TREE edges become BINDING: raising a leaf's last tree-edge weight by
# delta moves d(s,u) by ~delta (verified), so C3' tree-vs-non-tree is meaningful.
# Connectivity is guaranteed by a random spanning tree; extra random edges fill to
# the target average degree (avg_deg=4 is the sweet spot: decorrelated yet binding).
# ---------------------------------------------------------------------------
def make_weighted_graph(n: int, avg_deg: float, seed: int,
                        w_low: float = 0.2, w_high: float = 5.0):
    """Connected graph, N nodes, log-uniform random weights in [w_low, w_high]
    (strictly positive, topology-independent). Returns (A_weighted, dummy_coords)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    A = np.zeros((n, n))

    def setw(a, b):
        w = float(np.exp(rng.uniform(np.log(w_low), np.log(w_high))))
        A[a, b] = w
        A[b, a] = w

    # random spanning tree -> guarantees one connected component (all n nodes)
    for k in range(1, n):
        a = perm[k]
        b = perm[rng.randint(0, k)]
        setw(a, b)
    # extra random edges up to target average degree
    target_edges = int(avg_deg * n / 2)
    cur = int((A > 0).sum() // 2)
    tries = 0
    while cur < target_edges and tries < 20 * target_edges:
        a, b = int(rng.randint(0, n)), int(rng.randint(0, n))
        if a != b and A[a, b] == 0:
            setw(a, b)
            cur += 1
        tries += 1
    return torch.tensor(A, dtype=DTYPE), np.zeros((n, 2))


# ---------------------------------------------------------------------------
# Ground-truth SSSP via TWO independent methods (self-check S1).
# Returns: dist[u], predecessor pred[u] (parent on the SP-tree), and the SP-tree
# edge set as a frozenset of undirected (min,max) tuples; plus the per-node
# SP-PATH edge set: the edges on the s->u path.
# ---------------------------------------------------------------------------
def dijkstra_nx(Aw: Tensor, s: int):
    import networkx as nx

    n = Aw.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    A = Aw.cpu().numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]))
    dist, paths = nx.single_source_dijkstra(G, s, weight="weight")
    d = np.full(n, np.inf)
    pred = np.full(n, -1, dtype=int)
    for u, du in dist.items():
        d[u] = du
    for u, p in paths.items():
        if len(p) >= 2:
            pred[u] = p[-2]
    return d, pred, paths


def dijkstra_scipy(Aw: Tensor, s: int):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    A = Aw.cpu().numpy()
    Msp = csr_matrix(A)  # zeros are absent edges; positive weights present
    d, predecessors = sp_dijkstra(
        Msp, directed=False, indices=s, return_predecessors=True
    )
    pred = np.where(predecessors == -9999, -1, predecessors)
    return d, pred


def sp_tree_edges(pred: np.ndarray, s: int) -> set:
    """Undirected (min,max) edge set of the shortest-path tree."""
    E = set()
    for u in range(len(pred)):
        if u == s or pred[u] < 0:
            continue
        a, b = u, int(pred[u])
        E.add((min(a, b), max(a, b)))
    return E


def sp_path_edges(paths: dict, u: int) -> set:
    """Undirected edge set of the s->u shortest path."""
    if u not in paths:
        return set()
    p = paths[u]
    return {(min(p[k], p[k + 1]), max(p[k], p[k + 1])) for k in range(len(p) - 1)}


# ---------------------------------------------------------------------------
# Features: channel 0 = source one-hot, channel 1 = constant bias.
# Same convention as the CC gate (self-check S2 reuses x[s,0]).
# ---------------------------------------------------------------------------
def make_features(n: int, s: int) -> Tensor:
    X = torch.zeros(n, 2, dtype=DTYPE)
    X[s, 0] = 1.0
    X[:, 1] = 1.0
    return X


# ---------------------------------------------------------------------------
# Operator with UNNORMALIZED weighted adjacency: A_w[i,j] = w_ij directly, so
# dz*/dw_e is a literal S_c column. kappa is enforced on ||W||*||A_w||.
# We reuse IGNN_Kappa unchanged: its operator is relu(A_hat @ W(Z) + X_proj) and
# _project_W targets ||W|| = kappa / ||A_hat||. We pass A_w as "A_hat" and its
# spectral norm as A_hat_spectral_norm.
# ---------------------------------------------------------------------------
def normalize_weights(Aw: Tensor) -> Tensor:
    """Scale all weights by a constant so the spectral radius of the raw weighted
    adjacency is ~1 (keeps the kappa<1 budget meaningful; pure rescale, preserves
    the shortest-path TREE exactly -- argmin is scale-invariant)."""
    sr = float(torch.linalg.eigvalsh(Aw)[-1].abs())
    if sr < 1e-9:
        return Aw
    return Aw / sr


def operator(model: IGNN_Kappa, Z: Tensor, A_w: Tensor, X_proj: Tensor) -> Tensor:
    return F_func.relu(A_w @ model.W(Z) + X_proj)


def solve_equilibrium(model, A_w, X, max_iter=2000, tol=1e-10):
    X_proj = model.U(X)
    ctx = {"A_hat": A_w, "X_proj": X_proj}
    N = X.shape[0]
    Z = torch.zeros(N, model.hidden, dtype=A_w.dtype, device=A_w.device)
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = operator(model, Z, A_w, X_proj)
            if (Z_new - Z).norm() < tol * max(float(Z.norm()), 1.0):
                Z = Z_new
                break
            Z = Z_new
    return Z, ctx, X_proj


def frozen_mask(model, Z_star, A_w, X_proj):
    with torch.no_grad():
        pre = A_w @ model.W(Z_star) + X_proj
        return (pre > 0).to(Z_star.dtype)


def jz_kron(model, mask, A_w):
    """J_z = diag(vec mask) kron(A_w, W), row-major over (N,hidden).
    Bit-identical to compute_jacobian (verified in exp_reachability)."""
    W = model.W.weight  # (h,h); operator computes W(Z)=Z @ W^T
    K = torch.kron(A_w, W)
    return mask.reshape(-1).unsqueeze(1) * K


def jx_kron(model, mask):
    """J_x = dF/dvec(X) = diag(vec mask) kron(I_N, U), row-major over (N,hidden)."""
    U = model.U.weight  # (hidden, in_features)
    N = mask.shape[0]
    I_N = torch.eye(N, dtype=U.dtype, device=U.device)
    K = torch.kron(I_N, U)
    return mask.reshape(-1).unsqueeze(1) * K


def resolvent(J_z):
    D = J_z.shape[0]
    I = torch.eye(D, dtype=J_z.dtype, device=J_z.device)
    return torch.linalg.inv(I - J_z)


def spectral_radius_jz(J_z) -> float:
    return float(torch.linalg.eigvals(J_z).abs().max())


# ---------------------------------------------------------------------------
# Training: regress NORMALIZED shortest-path distances d(s,.) over a union of
# weighted graphs. kappa<1 enforced via model._project_W() after every step.
# Distances are min-max normalized per graph (target in [0,1]); the head is a
# single linear readout to one scalar (regression).
# ---------------------------------------------------------------------------
def build_bundle(n_graphs, n, avg_deg, seed):
    rng = random.Random(seed)
    graphs = []
    for g in range(n_graphs):
        Aw, _ = make_weighted_graph(n, avg_deg, seed=seed + 7919 * g)
        Aw = normalize_weights(Aw)
        m = Aw.shape[0]
        # source = max-degree node (deterministic, central -> rich tree)
        deg = (Aw > 0).sum(1)
        s = int(deg.argmax())
        d, pred, paths = dijkstra_nx(Aw, s)
        d_sci, pred_sci = dijkstra_scipy(Aw, s)
        # S1 self-check baked in: distances must match
        assert np.allclose(d, d_sci, atol=1e-9), "S1 distance mismatch"
        dn = d / (d.max() if d.max() > 0 else 1.0)  # normalized target in [0,1]
        graphs.append(
            {
                "A_w": Aw, "m": m, "s": s,
                "X": make_features(m, s),
                "dist": torch.tensor(d, dtype=DTYPE),
                "dist_norm": torch.tensor(dn, dtype=DTYPE),
                "pred": pred, "paths": paths,
            }
        )
    return graphs


def block_diag(mats):
    sizes = [m.shape[0] for m in mats]
    N = sum(sizes)
    out = torch.zeros(N, N, dtype=mats[0].dtype)
    off = 0
    for m in mats:
        sz = m.shape[0]
        out[off:off + sz, off:off + sz] = m
        off += sz
    return out


def train_sssp_ignn(train_graphs, device, seed, kappa, hidden=64, epochs=600):
    set_seed(seed)
    A_w = block_diag([g["A_w"] for g in train_graphs]).to(device)
    X = torch.cat([g["X"] for g in train_graphs], 0).to(device)
    yt = torch.cat([g["dist_norm"] for g in train_graphs], 0).to(device)
    N = A_w.shape[0]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(0.7 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:]] = True

    A_w_sn = float(torch.linalg.svdvals(A_w)[0])
    model = IGNN_Kappa(2, hidden=hidden, n_classes=1, kappa=kappa,
                       A_hat_spectral_norm=A_w_sn).to(device).to(DTYPE)

    def fwd(differentiable):
        X_proj = model.U(X)
        Z = torch.zeros(N, model.hidden, dtype=A_w.dtype, device=device)
        cm = torch.enable_grad() if differentiable else torch.no_grad()
        with cm:
            for _ in range(80):
                Z = operator(model, Z, A_w, X_proj)
        return model.head(Z).squeeze(-1)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        pred = fwd(True)
        loss = F_func.mse_loss(pred[train_mask], yt[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        model._project_W()
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                pv = fwd(False)
                vmse = float(F_func.mse_loss(pv[val_mask], yt[val_mask]))
            if vmse < best_val:
                best_val = vmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = fwd(False)

        def r2(mask):
            yy = yt[mask]
            pp = pv[mask]
            ss_res = float(((yy - pp) ** 2).sum())
            ss_tot = float(((yy - yy.mean()) ** 2).sum()) + 1e-12
            return 1.0 - ss_res / ss_tot

        def corr(mask):
            yy = yt[mask].cpu().numpy()
            pp = pv[mask].cpu().numpy()
            if yy.std() < 1e-9 or pp.std() < 1e-9:
                return 0.0
            return float(np.corrcoef(yy, pp)[0, 1])

    return model, {
        "train_r2": r2(train_mask), "val_r2": r2(val_mask),
        "train_corr": corr(train_mask), "val_corr": corr(val_mask),
        "val_mse": best_val,
    }


def train_unconstrained_control(train_graphs, eval_graphs, device, seed,
                                depth=15, hidden=128, epochs=600, kappa_big=3.0):
    """CONTROL for the wall diagnosis: SAME architecture, NO kappa projection,
    trained as a finite unroll of `depth` steps (== an explicit GNN of that depth;
    a converged DEQ is depth->inf). If even the UNCONSTRAINED/expansive model cannot
    learn SSSP, the failure is an ARCHITECTURAL expressivity floor (linear-aggregation
    relu operator lacks the min-plus bias), not specifically the kappa<1 contraction
    wall. Returns (train_corr, heldout_corr, ||W||*||A||)."""
    set_seed(seed)
    A_w = block_diag([g["A_w"] for g in train_graphs]).to(device)
    X = torch.cat([g["X"] for g in train_graphs], 0).to(device)
    yt = torch.cat([g["dist_norm"] for g in train_graphs], 0).to(device)
    N = A_w.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    ntr = int(0.7 * N)
    trm = torch.zeros(N, dtype=torch.bool)
    trm[perm[:ntr]] = True
    sn = float(torch.linalg.svdvals(A_w)[0])
    model = IGNN_Kappa(2, hidden, 1, kappa=kappa_big,
                       A_hat_spectral_norm=sn).to(device).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        Xp = model.U(X)
        Z = torch.zeros(N, hidden, dtype=DTYPE, device=device)
        for _ in range(depth):
            Z = F_func.relu(A_w @ model.W(Z) + Xp)
        pred = model.head(Z).squeeze(-1)
        loss = F_func.mse_loss(pred[trm], yt[trm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        # NO _project_W -> unconstrained
    model.eval()
    with torch.no_grad():
        Xp = model.U(X)
        Z = torch.zeros(N, hidden, dtype=DTYPE, device=device)
        for _ in range(depth):
            Z = F_func.relu(A_w @ model.W(Z) + Xp)
        pv = model.head(Z).squeeze(-1)
        ytr, ptr = yt[trm].cpu().numpy(), pv[trm].cpu().numpy()
        tcorr = float(np.corrcoef(ytr, ptr)[0, 1]) if ytr.std() > 1e-9 else 0.0
        hc = []
        for gg in eval_graphs:
            A = gg["A_w"].to(device)
            X2 = gg["X"].to(device)
            Xp2 = model.U(X2)
            Z2 = torch.zeros(gg["m"], hidden, dtype=DTYPE, device=device)
            for _ in range(depth):
                Z2 = F_func.relu(A @ model.W(Z2) + Xp2)
            yh = model.head(Z2).squeeze(-1).cpu().numpy()
            yy = gg["dist_norm"].cpu().numpy()
            if yy.std() > 1e-9 and yh.std() > 1e-9:
                hc.append(float(np.corrcoef(yy, yh)[0, 1]))
    prod = float(torch.linalg.svdvals(model.W.weight.detach())[0]) * sn
    return tcorr, float(np.mean(hc)) if hc else float("nan"), prod


# ---------------------------------------------------------------------------
# Per-node EDGE-WEIGHT sensitivity from the resolvent: ||dz*_u / dw_e|| for every
# edge e. This IS the constrained sensitivity column S_c[u-block, e]. We compute
# S = (I-J_z)^{-1} J_A via the bug-audited iem path, then S_c per undirected edge.
# ---------------------------------------------------------------------------
def resolvent_edge_sensitivity(model, A_w, X, hidden):
    Z_star, ctx, X_proj = solve_equilibrium(model, A_w, X)
    mask = frozen_mask(model, Z_star, A_w, X_proj)

    def F(z, c):
        return operator(model, z, c["A_hat"], c["X_proj"])

    ctx = {"A_hat": A_w, "X_proj": X_proj}
    # bug-audited finite-difference structural Jacobian (edges_only -> only real edges)
    J_z_fd, J_A, col_map = _compute_structural_jacobian(F, Z_star, ctx, "A_hat",
                                                        edges_only=True)
    S = structural_sensitivity_matrix(F, Z_star, ctx, "A_hat", J_z=J_z_fd, J_A=J_A)
    # build per-undirected-edge S_c from the edges_only columns
    N = A_w.shape[0]
    # map (i,j)->column index in S
    cidx = {(i, j): k for k, (i, j) in enumerate(col_map)}
    edge_list = []
    cols = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_w[i, j].abs() > 1e-10:
                col = S[:, cidx[(i, j)]] + S[:, cidx[(j, i)]]
                cols.append(col)
                edge_list.append((i, j))
    S_c = torch.stack(cols, dim=1) if cols else torch.zeros(S.shape[0], 0,
                                                            dtype=S.dtype, device=S.device)
    # per-node gain to edge e: norm over u's hidden block
    D = S_c.shape[0]
    Sc_nodes = S_c.reshape(N, hidden, S_c.shape[1])
    gain = Sc_nodes.norm(dim=1)  # (N, |E|) -> gain[u, e] = ||dz*_u/dw_e||
    return gain.detach().cpu().numpy(), edge_list, J_z_fd, Z_star, mask


def jz_for_eig(model, A_w, X, hidden):
    Z_star, ctx, X_proj = solve_equilibrium(model, A_w, X)
    mask = frozen_mask(model, Z_star, A_w, X_proj)
    return jz_kron(model, mask, A_w).detach()


# ---------------------------------------------------------------------------
# BASELINE 1: input-gradient ||d y_hat_u / d w_e|| through the FULL nonlinear solve
# (black-box). This is the "explain by gradient of the prediction wrt the weight"
# heuristic. Computed per node by backprop through the unrolled fixed point.
# ---------------------------------------------------------------------------
def inputgrad_edge_sensitivity(model, A_w, X, edge_list):
    A = A_w.clone().detach().requires_grad_(True)
    X_proj = model.U(X)
    N = X.shape[0]
    Z = torch.zeros(N, model.hidden, dtype=A_w.dtype, device=A_w.device)
    for _ in range(300):
        Z = F_func.relu(A @ model.W(Z) + X_proj)
    yhat = model.head(Z).squeeze(-1)  # predicted normalized distance per node
    gains = np.zeros((N, len(edge_list)))
    for u in range(N):
        gx = torch.autograd.grad(yhat[u], A, retain_graph=(u < N - 1))[0]
        for k, (i, j) in enumerate(edge_list):
            # undirected edge: sum |dy/dA_ij| + |dy/dA_ji|
            gains[u, k] = float(gx[i, j].abs() + gx[j, i].abs())
    return gains


# ---------------------------------------------------------------------------
# BASELINE 2: adjacency-incidence. Score 1 if edge e is incident to u, else 0.
# (the CC-style "support" baseline -- the definitional trap we want to BEAT.)
# BASELINE 3: graph-distance heuristic. Score edge e=(a,b) for node u by
# -(min hop-distance from {a,b} to u) i.e. closer edges rank higher (BFS hops on
# the unweighted support). A pure-topology heuristic ignorant of the learned model.
# ---------------------------------------------------------------------------
def adjacency_incidence_scores(edge_list, N):
    inc = np.zeros((N, len(edge_list)))
    for k, (i, j) in enumerate(edge_list):
        inc[i, k] = 1.0
        inc[j, k] = 1.0
    return inc


def hop_distance_scores(A_w, edge_list, N):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(N))
    A = A_w.cpu().numpy()
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                G.add_edge(i, j)
    hop = dict(nx.all_pairs_shortest_path_length(G))
    scores = np.zeros((N, len(edge_list)))
    for u in range(N):
        for k, (i, j) in enumerate(edge_list):
            hi = hop[u].get(i, 1e9)
            hj = hop[u].get(j, 1e9)
            scores[u, k] = -float(min(hi, hj))  # closer edge -> higher score
    return scores


# ---------------------------------------------------------------------------
# C1' scoring: per node u (u != s, reachable), label edges by membership in the
# s->u shortest-PATH edge set; rank by each method's score; report AUC. Average
# over nodes (only nodes with >=1 path edge and >=1 non-path edge -> AUC defined).
# ---------------------------------------------------------------------------
def auc_safe(scores, labels):
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    if roc_auc_score is not None:
        return float(roc_auc_score(labels, scores))
    # rank-based fallback
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def precision_at_k(scores, labels, k):
    if k <= 0 or labels.sum() == 0:
        return None
    top = np.argsort(-scores)[:k]
    return float(labels[top].sum() / k)


def c1_sptree_recovery(graph, method_scores: dict):
    """For each node u, AUC of each method's edge scores vs the s->u SP-path edges."""
    s = graph["s"]
    paths = graph["paths"]
    edge_list = graph["edge_list"]
    eidx = {e: k for k, e in enumerate(edge_list)}
    out = {m: {"auc": [], "p_at_path": []} for m in method_scores}
    for u in range(graph["m"]):
        if u == s or u not in paths:
            continue
        pe = sp_path_edges(paths, u)
        if not pe:
            continue
        labels = np.zeros(len(edge_list))
        for e in pe:
            if e in eidx:
                labels[eidx[e]] = 1.0
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        k_path = int(labels.sum())
        for m, sc in method_scores.items():
            a = auc_safe(sc[u], labels)
            if a is not None:
                out[m]["auc"].append(a)
            p = precision_at_k(sc[u], labels, k_path)
            if p is not None:
                out[m]["p_at_path"].append(p)
    return {m: {"auc": float(np.mean(v["auc"])) if v["auc"] else float("nan"),
                "p_at_path": float(np.mean(v["p_at_path"])) if v["p_at_path"] else float("nan"),
                "n_nodes": len(v["auc"])}
            for m, v in out.items()}


# ---------------------------------------------------------------------------
# C3' causal tree-vs-nontree. Pick the node u_far with the largest SP distance.
# A TREE edge = the last edge on u_far's path (pred[u_far], u_far): removing /
# increasing it MUST change d(s,u_far). A NON-TREE edge = an edge incident to
# u_far that is NOT on its path (a "shortcut that lost"). Perturb each by +delta on
# the weight; compare the resolvent-PREDICTED dz*_{u_far} to the RE-SOLVED nonlinear
# equilibrium AND to the true Dijkstra distance change.
# ---------------------------------------------------------------------------
def c3_tree_vs_nontree(model, graph, hidden, delta=0.05):
    A_w = graph["A_w"].to(DEVICE)
    X = graph["X"].to(DEVICE)
    s = graph["s"]
    pred = graph["pred"]
    paths = graph["paths"]
    # farthest reachable node
    d = graph["dist"].cpu().numpy()
    finite = np.where(np.isfinite(d))[0]
    u = int(finite[np.argmax(d[finite])])
    if pred[u] < 0:
        return None
    tree_edge = (min(u, int(pred[u])), max(u, int(pred[u])))
    path_e = sp_path_edges(paths, u)
    # a non-tree edge incident to u not on its path
    nontree_edge = None
    for v in range(graph["m"]):
        if A_w[u, v] > 1e-10:
            e = (min(u, v), max(u, v))
            if e not in path_e:
                nontree_edge = e
                break
    if nontree_edge is None:
        return None

    gain, edge_list, _, Z_star, mask = resolvent_edge_sensitivity(model, A_w, X, hidden)
    eidx = {e: k for k, e in enumerate(edge_list)}
    # resolvent S_c columns (signed, full vector) for predicted dz*
    J_z_fd, J_A, col_map = _compute_structural_jacobian(
        lambda z, c: operator(model, z, c["A_hat"], c["X_proj"]),
        Z_star, {"A_hat": A_w, "X_proj": model.U(X)}, "A_hat", edges_only=True)
    S = structural_sensitivity_matrix(
        lambda z, c: operator(model, z, c["A_hat"], c["X_proj"]),
        Z_star, {"A_hat": A_w, "X_proj": model.U(X)}, "A_hat", J_z=J_z_fd, J_A=J_A)
    cidx = {(i, j): k for k, (i, j) in enumerate(col_map)}
    N = A_w.shape[0]

    def predict_and_resolve(edge):
        i, j = edge
        col = (S[:, cidx[(i, j)]] + S[:, cidx[(j, i)]])  # dz*/dw_e
        pred_dz = (col * delta).detach()
        pred_du = float(pred_dz.reshape(N, hidden)[u].norm())
        # re-solve nonlinear equilibrium with the perturbed weight
        A2 = A_w.clone()
        A2[i, j] += delta
        A2[j, i] += delta
        Z2, _, _ = solve_equilibrium(model, A2, X)
        real_dz = (Z2 - Z_star).reshape(-1).detach()
        real_du = float((Z2 - Z_star)[u].norm())
        cos = float(pred_dz @ real_dz / (pred_dz.norm() * real_dz.norm() + 1e-12))
        # true algorithmic distance change (Dijkstra on perturbed weights)
        d2, _, _ = dijkstra_nx(A2.cpu(), s)
        true_du = float(d2[u] - graph["dist"][u].item())
        # model's predicted-distance change via head
        yhat0 = float(model.head(Z_star).squeeze(-1)[u])
        yhat2 = float(model.head(Z2).squeeze(-1)[u])
        return {
            "pred_du_norm": pred_du, "real_du_norm": real_du,
            "cos": cos, "true_dijkstra_du": true_du,
            "model_yhat_du": yhat2 - yhat0,
        }

    return {
        "u_far": u, "dist_u": float(d[u]),
        "tree_edge": tree_edge, "nontree_edge": nontree_edge,
        "tree": predict_and_resolve(tree_edge),
        "nontree": predict_and_resolve(nontree_edge),
    }


# ---------------------------------------------------------------------------
# C2' eigen-mode organization. Reduce top-k J_z eigvecs (ranked by resolvent gain
# 1/|1-lambda|) to node-space norm profiles; measure alignment with an SP-TREE
# structural signal: the per-node "tree depth" indicator (#hops to s on the tree).
# Control: A_w's own eigvecs (pure adjacency spectrum). If the resolvent modes align
# with tree structure MORE than the adjacency control, that is genuine tree-decoding.
# ---------------------------------------------------------------------------
def tree_depth_vector(graph) -> np.ndarray:
    pred = graph["pred"]
    s = graph["s"]
    depth = np.zeros(graph["m"])
    for u in range(graph["m"]):
        d, steps = 0.0, 0
        v = u
        while v != s and pred[v] >= 0 and steps < graph["m"]:
            d += 1.0
            v = int(pred[v])
            steps += 1
        depth[u] = d
    return depth


def node_profiles(eigvecs: Tensor, N: int, hidden: int) -> Tensor:
    k = eigvecs.shape[1]
    prof = torch.zeros(N, k, dtype=torch.float64)
    for c in range(k):
        v = eigvecs[:, c].reshape(N, hidden)
        prof[:, c] = v.norm(dim=1).double()
    return prof


def c2_eigen_alignment(J_z, A_w, graph, N, hidden, topk=6):
    depth = tree_depth_vector(graph)
    depth = depth - depth.mean()
    if np.linalg.norm(depth) < 1e-9:
        return {"resolvent": float("nan"), "adjacency": float("nan")}
    depth_u = depth / np.linalg.norm(depth)

    # resolvent-weighted J_z eigvecs
    evals, evecs = torch.linalg.eig(J_z)
    gain = 1.0 / (1.0 - evals).abs()
    order = torch.argsort(gain, descending=True)[:topk]
    V = evecs[:, order].real.double()
    prof = node_profiles(V, N, hidden).cpu().numpy()  # (N, topk)

    def best_align(profile_mat):
        best = 0.0
        for c in range(profile_mat.shape[1]):
            p = profile_mat[:, c] - profile_mat[:, c].mean()
            if np.linalg.norm(p) < 1e-9:
                continue
            p = p / np.linalg.norm(p)
            best = max(best, abs(float(p @ depth_u)))
        return best

    align_res = best_align(prof)

    # adjacency control: A_w eigvecs (node-space already)
    aw_evals, aw_evecs = torch.linalg.eigh(A_w)
    idx = torch.argsort(aw_evals.abs(), descending=True)[:topk]
    AV = aw_evecs[:, idx].real.double().cpu().numpy()  # (N, topk)
    align_adj = best_align(AV)

    return {"resolvent": align_res, "adjacency": align_adj}


# ===========================================================================
# Self-checks
# ===========================================================================
def self_check_dijkstra(verbose=True) -> bool:
    """S1: networkx vs scipy dijkstra agree on distances + tree predecessors."""
    Aw, _ = make_weighted_graph(40, avg_deg=4, seed=SEED)
    s = int((Aw > 0).sum(1).argmax())
    d_nx, pred_nx, _ = dijkstra_nx(Aw, s)
    d_sci, pred_sci = dijkstra_scipy(Aw, s)
    ok_d = np.allclose(d_nx, d_sci, atol=1e-9)
    # tree edge sets must match (predecessors can differ on weight ties; compare sets)
    e_nx = sp_tree_edges(pred_nx, s)
    e_sci = sp_tree_edges(pred_sci, s)
    ok_t = (e_nx == e_sci)
    if verbose:
        print(f"  [S1] dijkstra nx-vs-scipy: dist_match={ok_d} tree_match={ok_t} "
              f"(|tree|={len(e_nx)})", flush=True)
    return bool(ok_d and ok_t)


def self_check_resolvent_input(verbose=True) -> bool:
    """S2: resolvent block (I-J_z)^{-1} J_x[:,src] matches autograd dz*/dx_s
    on a tiny weighted graph (reuses the bit-exact CC path, weighted operator)."""
    torch.manual_seed(SEED)
    Aw, _ = make_weighted_graph(12, avg_deg=4, seed=11)
    Aw = normalize_weights(Aw)
    N = Aw.shape[0]
    Aw = Aw.to(DEVICE)
    s = int((Aw > 0).sum(1).argmax())
    X = make_features(N, s).to(DEVICE)
    model = IGNN_Kappa(2, 6, 1, kappa=0.7,
                       A_hat_spectral_norm=float(torch.linalg.svdvals(Aw)[0])
                       ).to(DEVICE).to(DTYPE)
    model._project_W()
    model.eval()
    Xr = X.clone().detach().requires_grad_(True)
    X_proj = model.U(Xr)
    Z = torch.zeros(N, model.hidden, dtype=DTYPE, device=DEVICE)
    for _ in range(400):
        Z = F_func.relu(Aw @ model.W(Z) + X_proj)
    D = Z.numel()
    Zf = Z.reshape(-1)
    ref = torch.zeros(D, dtype=DTYPE, device=DEVICE)
    for dd in range(D):
        gx = torch.autograd.grad(Zf[dd], Xr, retain_graph=True)[0]
        ref[dd] = gx[s, 0]
    Z_star, _, X_proj2 = solve_equilibrium(model, Aw, X)
    mask = frozen_mask(model, Z_star, Aw, X_proj2)
    J_z = jz_kron(model, mask, Aw)
    J_x = jx_kron(model, mask)
    R = resolvent(J_z)
    pred = (R @ J_x[:, s * 2 + 0]).detach()
    ref = ref.detach()
    err = float((pred - ref).norm() / (ref.norm() + 1e-12))
    cos = float((pred @ ref) / (pred.norm() * ref.norm() + 1e-12))
    ok = err < 1e-5
    if verbose:
        print(f"  [S2] resolvent block vs autograd dz*/dx_s: rel_err={err:.2e} "
              f"cos={cos:.6f} PASS={ok}", flush=True)
    return ok


def self_check_Sc_edge(verbose=True) -> bool:
    """S3: S_c column (I-J_z)^{-1}(J_A col) matches autograd dz*/dw_e for one edge
    (THE C1' path). Validates that ||dz*_u/dw_e|| we rank on is exact."""
    torch.manual_seed(SEED)
    Aw, _ = make_weighted_graph(12, avg_deg=4, seed=13)
    Aw = normalize_weights(Aw).to(DEVICE)
    N = Aw.shape[0]
    s = int((Aw > 0).sum(1).argmax())
    X = make_features(N, s).to(DEVICE)
    model = IGNN_Kappa(2, 6, 1, kappa=0.7,
                       A_hat_spectral_norm=float(torch.linalg.svdvals(Aw)[0])
                       ).to(DEVICE).to(DTYPE)
    model._project_W()
    model.eval()
    # pick an edge
    edge = None
    for i in range(N):
        for j in range(i + 1, N):
            if Aw[i, j] > 1e-10:
                edge = (i, j)
                break
        if edge:
            break
    i, j = edge

    # autograd reference: differentiable solve wrt a single scalar weight w on edge
    w = torch.tensor(float(Aw[i, j]), dtype=DTYPE, device=DEVICE, requires_grad=True)
    A2 = Aw.clone()
    A2[i, j] = w
    A2[j, i] = w
    X_proj = model.U(X)
    Z = torch.zeros(N, model.hidden, dtype=DTYPE, device=DEVICE)
    for _ in range(500):
        Z = F_func.relu(A2 @ model.W(Z) + X_proj)
    Zf = Z.reshape(-1)
    D = Zf.numel()
    ref = torch.zeros(D, dtype=DTYPE, device=DEVICE)
    for dd in range(D):
        gx = torch.autograd.grad(Zf[dd], w, retain_graph=True)[0]
        ref[dd] = gx
    ref = ref.detach()

    # resolvent S_c column for edge (i,j): S[:,iN+j]+S[:,jN+i]
    Z_star, ctx, X_proj2 = solve_equilibrium(model, Aw, X)

    def F(z, c):
        return operator(model, z, c["A_hat"], c["X_proj"])

    J_z_fd, J_A, col_map = _compute_structural_jacobian(
        F, Z_star, {"A_hat": Aw, "X_proj": X_proj2}, "A_hat", edges_only=True)
    S = structural_sensitivity_matrix(
        F, Z_star, {"A_hat": Aw, "X_proj": X_proj2}, "A_hat", J_z=J_z_fd, J_A=J_A)
    cidx = {(a, b): k for k, (a, b) in enumerate(col_map)}
    col = (S[:, cidx[(i, j)]] + S[:, cidx[(j, i)]]).detach()
    err = float((col - ref).norm() / (ref.norm() + 1e-12))
    cos = float((col @ ref) / (col.norm() * ref.norm() + 1e-12))
    ok = err < 1e-3  # finite-difference J_A -> looser than exact-kron S2
    if verbose:
        print(f"  [S3] S_c column vs autograd dz*/dw_e: rel_err={err:.2e} "
              f"cos={cos:.6f} PASS={ok}", flush=True)
    return ok


# ===========================================================================
# Main gate
# ===========================================================================
def ms(xs):
    a = np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def run_gate():
    t0 = time.time()
    print("=" * 78, flush=True)
    print("DECISIVE INTERP GATE — SSSP shortest-path-tree resolvent decoding (seed 42)",
          flush=True)
    print(f"device={DEVICE}", flush=True)
    print("=" * 78, flush=True)

    # ---- self-checks (abort if any fail)
    print("[self-checks]", flush=True)
    s1 = self_check_dijkstra()
    s2 = self_check_resolvent_input()
    s3 = self_check_Sc_edge()
    if not (s1 and s2 and s3):
        print("SELF-CHECK FAILED — aborting (machinery not trustworthy).", flush=True)
        sys.exit(1)

    # ---- train ONE kappa<1 IGNN on a union of weighted SSSP graphs (G1)
    KAPPA = 0.95
    N_PER = 60
    AVG_DEG = 4  # decorrelated-yet-binding regime (see make_weighted_graph note)
    print(f"\n[G1] training kappa<1 IGNN on SSSP (kappa={KAPPA}, N~={N_PER}/graph, "
          f"avg_deg={AVG_DEG})...", flush=True)
    train_graphs = build_bundle(n_graphs=12, n=N_PER, avg_deg=AVG_DEG, seed=SEED)

    # design validation: SSSP must NOT be reducible to hop-distance (else it is the
    # CC definitional trap in disguise). Report corr(weighted-dist, hop-dist).
    import networkx as _nx
    _cors = []
    for _g in train_graphs:
        _A = _g["A_w"].cpu().numpy()
        _G = _nx.Graph(); _G.add_nodes_from(range(_g["m"]))
        for _i in range(_g["m"]):
            for _j in range(_i + 1, _g["m"]):
                if _A[_i, _j] > 0:
                    _G.add_edge(_i, _j)
        _hop = np.array([_nx.shortest_path_length(_G, _g["s"], _u)
                         for _u in range(_g["m"])], float)
        _d = _g["dist"].cpu().numpy()
        _cors.append(float(np.corrcoef(_d, _hop)[0, 1]))
    print(f"  [design] corr(weighted-dist, hop-dist)={np.mean(_cors):.3f} "
          f"(want << 0.9 so SSSP is NOT reducible to BFS/hop -- non-definitional)",
          flush=True)

    model, tr = train_sssp_ignn(train_graphs, DEVICE, SEED, KAPPA, hidden=64, epochs=600)
    print(f"  train: R2={tr['train_r2']:.4f} corr={tr['train_corr']:.4f} | "
          f"val(held-out NODES): R2={tr['val_r2']:.4f} corr={tr['val_corr']:.4f}",
          flush=True)

    # held-out GRAPHS for the resolvent gates + held-out-graph G1 accuracy
    eval_graphs = build_bundle(n_graphs=8, n=N_PER, avg_deg=AVG_DEG, seed=SEED + 50_000)

    # ---- UNCONSTRAINED control (separates contraction-wall from architecture-floor)
    print("[G1-control] training UNCONSTRAINED same-architecture (no kappa projection, "
          "depth-15 unroll)...", flush=True)
    uc_train, uc_held, uc_prod = train_unconstrained_control(
        train_graphs, eval_graphs, DEVICE, SEED, depth=15, hidden=128, epochs=600)
    print(f"  unconstrained: train_corr={uc_train:.4f} heldout_corr={uc_held:.4f} "
          f"||W||*||A||={uc_prod:.3f}  "
          f"(if held-out ALSO low -> architectural floor, not the kappa<1 wall)",
          flush=True)

    rows = []
    g1_corr, g1_mae, kappas = [], [], []
    c1 = {m: {"auc": [], "p": []} for m in
          ["resolvent", "inputgrad", "adjacency", "hopdist"]}
    c3_tree_pred, c3_tree_real, c3_tree_cos = [], [], []
    c3_nontree_pred, c3_nontree_real, c3_nontree_cos = [], [], []
    c3_tree_true_dij, c3_nontree_true_dij = [], []
    c2_res, c2_adj = [], []

    print("\n[held-out graphs] evaluating G1 / C1' / C3' / C2' ...", flush=True)
    for gi, graph in enumerate(eval_graphs):
        A_w = graph["A_w"].to(DEVICE)
        X = graph["X"].to(DEVICE)
        hidden = model.hidden
        m = graph["m"]

        # --- G1 held-out-graph distance accuracy ---
        Z_star, _, _ = solve_equilibrium(model, A_w, X)
        yhat = model.head(Z_star).squeeze(-1).detach().cpu().numpy()
        ytrue = graph["dist_norm"].cpu().numpy()
        if ytrue.std() > 1e-9 and yhat.std() > 1e-9:
            g1_corr.append(float(np.corrcoef(ytrue, yhat)[0, 1]))
        g1_mae.append(float(np.abs(ytrue - yhat).mean()))

        # --- kappa = rho(J_z) ---
        J_z = jz_for_eig(model, A_w, X, hidden)
        kap = spectral_radius_jz(J_z)
        kappas.append(kap)

        # --- resolvent edge sensitivity (the S_c path) ---
        gain, edge_list, _, _, _ = resolvent_edge_sensitivity(model, A_w, X, hidden)
        graph["edge_list"] = edge_list

        # --- baselines ---
        ig = inputgrad_edge_sensitivity(model, A_w, X, edge_list)
        inc = adjacency_incidence_scores(edge_list, m)
        hop = hop_distance_scores(A_w, edge_list, m)

        # --- C1' SP-tree recovery per node ---
        method_scores = {
            "resolvent": gain, "inputgrad": ig,
            "adjacency": inc, "hopdist": hop,
        }
        rec = c1_sptree_recovery(graph, method_scores)
        for mth in c1:
            if np.isfinite(rec[mth]["auc"]):
                c1[mth]["auc"].append(rec[mth]["auc"])
            if np.isfinite(rec[mth]["p_at_path"]):
                c1[mth]["p"].append(rec[mth]["p_at_path"])

        # --- C3' tree-vs-nontree ---
        c3 = c3_tree_vs_nontree(model, graph, hidden, delta=0.05)
        if c3 is not None:
            c3_tree_pred.append(c3["tree"]["pred_du_norm"])
            c3_tree_real.append(c3["tree"]["real_du_norm"])
            c3_tree_cos.append(c3["tree"]["cos"])
            c3_tree_true_dij.append(abs(c3["tree"]["true_dijkstra_du"]))
            c3_nontree_pred.append(c3["nontree"]["pred_du_norm"])
            c3_nontree_real.append(c3["nontree"]["real_du_norm"])
            c3_nontree_cos.append(c3["nontree"]["cos"])
            c3_nontree_true_dij.append(abs(c3["nontree"]["true_dijkstra_du"]))

        # --- C2' eigen alignment ---
        c2 = c2_eigen_alignment(J_z, A_w, graph, m, hidden, topk=6)
        if np.isfinite(c2["resolvent"]):
            c2_res.append(c2["resolvent"])
        if np.isfinite(c2["adjacency"]):
            c2_adj.append(c2["adjacency"])

        rows.append({
            "graph": gi, "m": m, "n_edges": len(edge_list),
            "kappa_rho_Jz": kap,
            "g1_corr": g1_corr[-1] if g1_corr else float("nan"),
            "g1_mae": g1_mae[-1],
            "c1_auc_resolvent": rec["resolvent"]["auc"],
            "c1_auc_inputgrad": rec["inputgrad"]["auc"],
            "c1_auc_adjacency": rec["adjacency"]["auc"],
            "c1_auc_hopdist": rec["hopdist"]["auc"],
            "c1_p_resolvent": rec["resolvent"]["p_at_path"],
            "c1_p_inputgrad": rec["inputgrad"]["p_at_path"],
            "c1_p_adjacency": rec["adjacency"]["p_at_path"],
            "c1_p_hopdist": rec["hopdist"]["p_at_path"],
            "c3_tree_pred": c3["tree"]["pred_du_norm"] if c3 else float("nan"),
            "c3_tree_real": c3["tree"]["real_du_norm"] if c3 else float("nan"),
            "c3_tree_cos": c3["tree"]["cos"] if c3 else float("nan"),
            "c3_nontree_pred": c3["nontree"]["pred_du_norm"] if c3 else float("nan"),
            "c3_nontree_real": c3["nontree"]["real_du_norm"] if c3 else float("nan"),
            "c3_nontree_cos": c3["nontree"]["cos"] if c3 else float("nan"),
            "c2_resolvent": c2["resolvent"],
            "c2_adjacency": c2["adjacency"],
        })
        print(f"  graph {gi}: m={m} |E|={len(edge_list)} rho(Jz)={kap:.3f} "
              f"g1_corr={rows[-1]['g1_corr']:.3f} "
              f"C1'AUC res={rec['resolvent']['auc']:.3f} "
              f"ig={rec['inputgrad']['auc']:.3f} adj={rec['adjacency']['auc']:.3f} "
              f"hop={rec['hopdist']['auc']:.3f}", flush=True)

    # ---- aggregate
    g1c_m, g1c_s = ms(g1_corr)
    g1e_m, g1e_s = ms(g1_mae)
    kap_m, kap_s = ms(kappas)
    print("\n" + "=" * 78, flush=True)
    print("AGGREGATE (held-out graphs, seed 42)", flush=True)
    print("=" * 78, flush=True)
    print(f"kappa = rho(J_z) = {kap_m:.4f} +/- {kap_s:.4f}  "
          f"(<1 contraction {'HOLDS' if kap_m < 1 else 'VIOLATED'})", flush=True)
    print(f"[G1] held-out-graph distance: corr={g1c_m:.4f}+/-{g1c_s:.4f}  "
          f"MAE={g1e_m:.4f}+/-{g1e_s:.4f}  "
          f"(train R2={tr['train_r2']:.3f}, val-node R2={tr['val_r2']:.3f})", flush=True)

    print("\n[C1'] SP-TREE RECOVERY — AUC of edge-ranking vs s->u shortest-path edges:",
          flush=True)
    c1_summ = {}
    for mth in ["resolvent", "inputgrad", "adjacency", "hopdist"]:
        am, asd = ms(c1[mth]["auc"])
        pm, psd = ms(c1[mth]["p"])
        c1_summ[mth] = (am, asd, pm, psd)
        print(f"    {mth:10s}: AUC={am:.4f}+/-{asd:.4f}  P@|path|={pm:.4f}+/-{psd:.4f}",
              flush=True)
    res_auc = c1_summ["resolvent"][0]
    best_base = max(c1_summ["inputgrad"][0], c1_summ["adjacency"][0],
                    c1_summ["hopdist"][0])
    beats = res_auc - best_base
    print(f"    --> resolvent AUC {res_auc:.4f}  vs  best-baseline {best_base:.4f}  "
          f"(margin {beats:+.4f})", flush=True)

    print("\n[C3'] CAUSAL TREE vs NON-TREE — distance change at u_far:", flush=True)
    tp_m, _ = ms(c3_tree_pred); tr_m, _ = ms(c3_tree_real)
    np_m, _ = ms(c3_nontree_pred); nr_m, _ = ms(c3_nontree_real)
    tdij_m, _ = ms(c3_tree_true_dij); ndij_m, _ = ms(c3_nontree_true_dij)
    tcos_m, _ = ms(c3_tree_cos); ncos_m, _ = ms(c3_nontree_cos)
    print(f"    TREE edge    : resolvent-pred |dz_u|={tp_m:.4e}  re-solved={tr_m:.4e}  "
          f"cos={tcos_m:.3f}  true-Dijkstra |dd|={tdij_m:.4e}", flush=True)
    print(f"    NON-TREE edge: resolvent-pred |dz_u|={np_m:.4e}  re-solved={nr_m:.4e}  "
          f"cos={ncos_m:.3f}  true-Dijkstra |dd|={ndij_m:.4e}", flush=True)
    sep_pred = tp_m / (np_m + 1e-12)
    sep_real = tr_m / (nr_m + 1e-12)
    sep_dij = tdij_m / (ndij_m + 1e-12)
    print(f"    separation tree/non-tree: resolvent-pred {sep_pred:.2f}x  "
          f"re-solved {sep_real:.2f}x  true-Dijkstra {sep_dij:.2f}x", flush=True)

    print("\n[C2'] EIGEN-MODE alignment with SP-tree depth (vs adjacency control):",
          flush=True)
    c2r_m, c2r_s = ms(c2_res); c2a_m, c2a_s = ms(c2_adj)
    print(f"    resolvent modes={c2r_m:.4f}+/-{c2r_s:.4f}  "
          f"adjacency modes={c2a_m:.4f}+/-{c2a_s:.4f}  "
          f"margin={c2r_m - c2a_m:+.4f}", flush=True)

    # ---- VERDICT
    print("\n" + "=" * 78, flush=True)
    contraction_ok = kap_m < 1.0
    # G1: does the kappa<1 model learn SSSP well enough that decoding is meaningful?
    learns_sssp = (g1c_m > 0.85) and (tr["val_r2"] > 0.6)
    # unconstrained control already printed above as (uc_train, uc_held, uc_prod)
    unconstrained_also_fails = (uc_held < 0.7)
    # "beats baselines" on the discriminating recovery task
    beats_baselines = (res_auc > best_base + 0.03) and (res_auc > 0.65)
    # causal tree/non-tree separation, matching the re-solved equilibrium
    clean_causal = (sep_pred > 2.0) and (sep_real > 2.0)

    if not learns_sssp:
        # the model never executed SSSP -> resolvent-decoding question is MOOT, and the
        # diagnosis splits on whether removing the contraction would have rescued it.
        if contraction_ok and not unconstrained_also_fails:
            verdict = ("CONTRACTION-WALL — kappa<1 IGNN CANNOT learn SSSP "
                       f"(held-out corr={g1c_m:.3f}, val-R2={tr['val_r2']:.3f}, "
                       f"rho(Jz)={kap_m:.2f}), but the UNCONSTRAINED same-architecture "
                       f"control DOES (held-out corr={uc_held:.3f}); the "
                       "expressivity<->contraction tension BITES. Ties AEGIS criticality: "
                       "long-range min-plus computation needs rho->1. C1'/C3' MOOT.")
        else:
            verdict = ("EXPRESSIVITY-FLOOR (architecture, not contraction) — NEITHER the "
                       f"kappa<1 IGNN (held-out corr={g1c_m:.3f}) NOR the unconstrained "
                       f"same-architecture control (held-out corr={uc_held:.3f}, "
                       f"||W||*||A||={uc_prod:.1f}) learns SSSP. The linear-aggregation "
                       "relu operator lacks the min-plus inductive bias for shortest "
                       "paths; under kappa<1 it further collapses to a strong contraction "
                       f"(rho(Jz)={kap_m:.2f}) that only smooths LOCAL distance. SSSP is "
                       "not executed, so resolvent SP-tree decoding (C1'/C3') is MOOT; "
                       "reported below for completeness. PIVOT DEAD on this algorithm.")
    elif beats_baselines and clean_causal:
        verdict = ("RESOLVENT-DECODES-SSSP — resolvent BEATS baselines on SP-tree "
                   f"recovery (AUC {res_auc:.3f} vs {best_base:.3f}, margin {beats:+.3f}) "
                   f"AND clean causal tree/non-tree separation "
                   f"(pred {sep_pred:.1f}x, re-solved {sep_real:.1f}x). PIVOT ALIVE.")
    elif beats_baselines:
        verdict = ("RESOLVENT-DECODES-SSSP (partial) — resolvent BEATS baselines on "
                   f"SP-tree recovery (AUC {res_auc:.3f} vs {best_base:.3f}) but causal "
                   f"separation weak (pred {sep_pred:.1f}x). Pivot promising; verify C3'.")
    else:
        verdict = ("WEAK — resolvent does NOT beat baselines on SP-tree recovery "
                   f"(AUC {res_auc:.3f} vs best-baseline {best_base:.3f}, "
                   f"margin {beats:+.3f}); ties as in CC. PIVOT DEAD.")
    print("VERDICT:", verdict, flush=True)
    print("=" * 78, flush=True)
    print(f"(elapsed {time.time() - t0:.1f}s)", flush=True)

    # ---- CSV
    out_csv = ROOT / "results" / "exp_interp_sssp.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}", flush=True)

    return {
        "kappa": (kap_m, kap_s),
        "g1_corr": (g1c_m, g1c_s), "g1_mae": (g1e_m, g1e_s),
        "train_r2": tr["train_r2"], "val_r2": tr["val_r2"],
        "c1": c1_summ, "res_auc": res_auc, "best_base": best_base, "beats": beats,
        "c3": {
            "tree_pred": tp_m, "tree_real": tr_m, "tree_cos": tcos_m,
            "nontree_pred": np_m, "nontree_real": nr_m, "nontree_cos": ncos_m,
            "tree_true_dij": tdij_m, "nontree_true_dij": ndij_m,
            "sep_pred": sep_pred, "sep_real": sep_real, "sep_dij": sep_dij,
        },
        "c2": {"resolvent": c2r_m, "adjacency": c2a_m, "margin": c2r_m - c2a_m},
        "control": {"uc_train": uc_train, "uc_held": uc_held, "uc_prod": uc_prod},
        "verdict": verdict,
        "selfchecks": {"s1": s1, "s2": s2, "s3": s3},
    }


if __name__ == "__main__":
    run_gate()
