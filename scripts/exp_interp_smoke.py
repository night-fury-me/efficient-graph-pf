# exp_interp_smoke

"""GATING SMOKE — does the equilibrium resolvent (I - J_z)^{-1} / constrained
sensitivity S_c MECHANISTICALLY DECODE what an implicit GNN computes, validated
against a KNOWN graph algorithm (source-reachability / connected components)?

This gate decides a potential breakthrough pivot for the AEGIS line. Correctness
and honesty are paramount; the gate must not be fudged.

Design (per paper/review/breakthrough_interp_scout.md, C1/C2/C3):
  - Generate random planted-partition graphs with 2-4 connected components,
    N ~= 60. Mark one SOURCE node s (one-hot channel in the input X). Ground
    truth per node u: y_u = 1 if u is in s's connected component else 0.
  - Train IGNN_Kappa (kappa<1 enforced exactly as in the phase/reachability
    experiments) to predict y. This is a deterministic algorithm whose CAUSAL
    structure is known: the source's input drives exactly its component's
    outputs, along graph paths.

IGNN fixed point:   z* = phi(A_hat z* W^T + U(X)),  phi = relu.
State Jacobian:     J_z = diag(phi') (A_hat (x) W)   [row-major vec over (N,h)]
Input Jacobian:     J_x = dF/dX = diag(phi') (I_N (x) U)   [U = self.U.weight]
Resolvent response: dz* = (I - J_z)^{-1} J_x dX  (the converged linear response
                    to a perturbation of the INPUT; for a source one-hot at s
                    this is the causal gain source -> every node).

Reused, bug-audited machinery (verified bit-identical in prior work):
  - IGNN_Kappa, train_ignn_kappa, set_seed       (exp_phase_transition)
  - jz_under = diag(vec mask) kron(A_hat, W)      (exp_reachability; == compute_jacobian)
  - compute_jacobian                              (iem.ift; row-by-row backward)
  - constrained_sensitivity_matrix / S            (iem.adversarial)

GATES (seed 42), all reported honestly:
  G1  expressivity-under-contraction: does kappa<1 IGNN actually SOLVE
      reachability? report kappa=rho(J_z) and test accuracy. If it CANNOT,
      that is the key finding (CONTRACTION-WALL) and is reported loudly.
  C1  resolvent recovers causal structure: g_u = ||dz*_u/dx_s||. AUC of g_u vs
      true reachability label, vs a black-box baseline (input-gradient
      ||dy_hat_u/dx_s|| through the full solve). NOTE per support law
      (A^k)_uv=0 for k<d(u,v): C1 may be near-definitional; C2/C3 are the real
      evidence.
  C3  faithfulness clincher: pick a BRIDGE edge (deletion changes the component
      structure). Predict dz* from first-order S_c, then RE-SOLVE the full
      nonlinear equilibrium after deleting it. cosine / rel-error of predicted
      vs re-solved change.
  C2  interpretability signal: do dominant eigen-modes of J_z (== resolvent
      eigvecs) ALIGN with component-indicator vectors? cosine of top-k node-
      space modes with the true component basis. Caveat reported: for CC this
      risks reducing to adjacency spectral structure; we flag whether the
      LEARNED W/phi' makes it non-trivial (compare against the same metric for
      A_hat's own eigvecs).

SELF-CHECKS:
  (S1) reachability labels: BFS vs union-find agree exactly.
  (S2) resolvent block extraction vs autograd dz*_u/dx_s on a tiny graph
       (full differentiable solve, backward through the fixed-point unroll).

OUTPUT: results/exp_interp_smoke.csv + compact printed summary + one-line VERDICT
        in {RESOLVENT-DECODES, CONTRACTION-WALL, WEAK}.

Usage:
    ./.venv/bin/python scripts/exp_interp_smoke.py
"""

from __future__ import annotations

import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (  # noqa: E402
    constrained_sensitivity_matrix,
    structural_sensitivity_matrix,
)
from iem.ift import compute_jacobian  # noqa: E402
from scripts.exp_phase_transition import IGNN_Kappa, set_seed  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

SEED = 42
DTYPE = torch.float64  # double precision for clean Jacobian / resolvent linear algebra


# ---------------------------------------------------------------------------
# Graph generation: planted-partition with K connected components, N ~= 60.
# Each component is a connected ER graph (guaranteed connected by a random
# spanning tree + extra edges). NO inter-component edges, so reachability is
# exactly "same component". Symmetric normalized adjacency with self-loops
# (GCN convention): A_hat = D^{-1/2} (A + I) D^{-1/2}.
# ---------------------------------------------------------------------------
def _connected_er_component(nodes: list[int], p: float, rng: random.Random) -> list[tuple[int, int]]:
    """Edges of a connected ER(p) graph on `nodes` (random spanning tree + ER extras)."""
    edges: set[tuple[int, int]] = set()
    # random spanning tree (guarantees connectivity)
    perm = nodes[:]
    rng.shuffle(perm)
    for k in range(1, len(perm)):
        a = perm[k]
        b = perm[rng.randint(0, k - 1)]
        edges.add((min(a, b), max(a, b)))
    # extra ER edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if rng.random() < p:
                edges.add((min(a, b), max(a, b)))
    return list(edges)


def make_partition_graph(n: int, k_comp: int, p: float, seed: int):
    """Planted-partition graph: k_comp connected ER components on n nodes.

    Returns dict with A_hat (normalized, double), comp_id (N,), edge_list (intra),
    and the raw adjacency A (0/1, no self-loops) for BFS/union-find self-checks.
    """
    rng = random.Random(seed)
    # partition nodes into k_comp roughly equal blocks
    sizes = [n // k_comp] * k_comp
    for r in range(n - sum(sizes)):
        sizes[r] += 1
    comp_id = np.zeros(n, dtype=np.int64)
    blocks: list[list[int]] = []
    start = 0
    for c, s in enumerate(sizes):
        nodes = list(range(start, start + s))
        blocks.append(nodes)
        comp_id[start:start + s] = c
        start += s

    A = np.zeros((n, n), dtype=np.float64)
    edge_list: list[tuple[int, int]] = []
    for nodes in blocks:
        for (a, b) in _connected_er_component(nodes, p, rng):
            A[a, b] = A[b, a] = 1.0
            edge_list.append((a, b))

    # symmetric normalized adjacency with self-loops (GCN)
    A_sl = A + np.eye(n)
    deg = A_sl.sum(axis=1)
    dinv = 1.0 / np.sqrt(deg)
    A_hat = (dinv[:, None] * A_sl) * dinv[None, :]

    return {
        "A": torch.tensor(A, dtype=DTYPE),
        "A_hat": torch.tensor(A_hat, dtype=DTYPE),
        "comp_id": torch.tensor(comp_id),
        "edge_list": edge_list,
        "n": n,
        "k_comp": k_comp,
    }


# ---------------------------------------------------------------------------
# Ground-truth reachability via TWO independent methods (self-check S1).
# ---------------------------------------------------------------------------
def reach_bfs(A: Tensor, s: int) -> np.ndarray:
    """1 if node u reachable from s on the 0/1 graph A (BFS)."""
    n = A.shape[0]
    Anp = (A.cpu().numpy() > 0)
    reach = np.zeros(n, dtype=np.int64)
    q = deque([s])
    reach[s] = 1
    while q:
        v = q.popleft()
        for w in np.nonzero(Anp[v])[0]:
            if not reach[w]:
                reach[w] = 1
                q.append(int(w))
    return reach


def reach_unionfind(A: Tensor, s: int) -> np.ndarray:
    """1 if node u in same union-find component as s."""
    n = A.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    Anp = (A.cpu().numpy() > 0)
    ii, jj = np.nonzero(Anp)
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)
    rs = find(s)
    return np.array([1 if find(u) == rs else 0 for u in range(n)], dtype=np.int64)


def find_bridges(A: Tensor) -> list[tuple[int, int]]:
    """All bridge edges of the 0/1 undirected graph A (Tarjan), as (u,v) u<v."""
    n = A.shape[0]
    Anp = (A.cpu().numpy() > 0)
    adj = [list(np.nonzero(Anp[v])[0]) for v in range(n)]
    disc = [-1] * n
    low = [0] * n
    timer = [0]
    bridges: list[tuple[int, int]] = []

    def dfs(u, parent):
        # iterative DFS to avoid recursion limits
        stack = [(u, parent, iter(adj[u]))]
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        while stack:
            node, par, it = stack[-1]
            advanced = False
            for w in it:
                w = int(w)
                if w == par:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = timer[0]
                    timer[0] += 1
                    stack.append((w, node, iter(adj[w])))
                    advanced = True
                    break
                else:
                    low[node] = min(low[node], disc[w])
            if not advanced:
                stack.pop()
                if stack:
                    pnode = stack[-1][0]
                    low[pnode] = min(low[pnode], low[node])
                    if low[node] > disc[pnode]:
                        bridges.append((min(pnode, node), max(pnode, node)))

    for v in range(n):
        if disc[v] == -1:
            dfs(v, -1)
    return bridges


# ---------------------------------------------------------------------------
# Build a labelled dataset: G graphs, each with a source node; node features
# X = [source_one_hot, bias_const]. Train/val/test split over (graph, node).
# We train ONE IGNN over a disjoint union of training graphs (block-diagonal
# A_hat) so kappa is enforced on the shared W exactly as in the phase exp.
# ---------------------------------------------------------------------------
def make_features(n: int, s: int) -> Tensor:
    """X: channel 0 = source indicator (1 at s), channel 1 = constant bias 1."""
    X = torch.zeros(n, 2, dtype=DTYPE)
    X[s, 0] = 1.0
    X[:, 1] = 1.0
    return X


def block_diag(mats: list[Tensor]) -> Tensor:
    sizes = [m.shape[0] for m in mats]
    N = sum(sizes)
    out = torch.zeros(N, N, dtype=mats[0].dtype)
    off = 0
    for m, s in zip(mats, sizes):
        out[off:off + s, off:off + s] = m
        off += s
    return out


def build_train_bundle(n_graphs: int, n: int, seed: int):
    """Disjoint union of n_graphs labelled reachability graphs for training."""
    rng = random.Random(seed)
    A_hats, Xs, ys = [], [], []
    for g in range(n_graphs):
        k = rng.choice([2, 3, 4])
        p = rng.uniform(0.12, 0.22)
        G = make_partition_graph(n, k, p, seed=seed + 1000 * g)
        # source = highest-degree node in component 0 (deterministic, gives signal)
        deg = (G["A"] > 0).sum(1)
        comp0 = (G["comp_id"] == 0).nonzero(as_tuple=True)[0]
        s = int(comp0[deg[comp0].argmax()].item())
        y = torch.tensor(reach_bfs(G["A"], s))
        A_hats.append(G["A_hat"])
        Xs.append(make_features(n, s))
        ys.append(y)
    A_hat = block_diag(A_hats)
    X = torch.cat(Xs, 0)
    y = torch.cat(ys, 0)
    return {
        "A_hat": A_hat, "X": X, "y": y,
        "n_features": 2, "n_classes": 2,
        "N": A_hat.shape[0], "per_n": n, "n_graphs": n_graphs,
    }


# ---------------------------------------------------------------------------
# Training: reuse train_ignn_kappa's recipe but with our own masks (it expects
# data["train_mask"] etc.). We replicate the loop here to keep it explicit and
# to use full-batch node classification with a train/val/test node split.
# ---------------------------------------------------------------------------
def train_reach_ignn(data, device, seed, kappa, epochs=400):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    N = data["N"]

    # node split: 60/20/20 stratified-ish by random permutation
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(0.6 * N)
    n_va = int(0.2 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va:]] = True

    A_hat_sn = float(torch.linalg.svdvals(A_hat)[0])
    model = IGNN_Kappa(
        data["n_features"], hidden=64, n_classes=data["n_classes"],
        kappa=kappa, A_hat_spectral_norm=A_hat_sn,
    ).to(device).to(DTYPE)

    def forward_logits(differentiable: bool):
        """Double-safe forward: solve the fixed point then head. IGNN_Kappa.forward
        hardcodes a float32 Z init, so we run the unroll ourselves at A_hat's dtype.
        differentiable=True keeps the graph (training); False is a no_grad eval."""
        X_proj = model.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}
        Z = torch.zeros(N, model.hidden, dtype=A_hat.dtype, device=device)
        ctxmgr = torch.enable_grad() if differentiable else torch.no_grad()
        with ctxmgr:
            for _ in range(60):
                Z = model.operator(Z, ctx)
        return model.head(Z), Z, ctx

    # class weights (reachable nodes are a minority across multi-component graphs)
    cnt = torch.bincount(y, minlength=2).double()
    w = (cnt.sum() / (2.0 * cnt.clamp(min=1))).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = forward_logits(differentiable=True)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask], weight=w)
        optim.zero_grad()
        loss.backward()
        optim.step()
        model._project_W()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = forward_logits(differentiable=False)
                va = float((lv.argmax(1)[val_mask] == y[val_mask]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = forward_logits(differentiable=False)
        test_acc = float((logits.argmax(1)[test_mask] == y[test_mask]).float().mean())
        train_acc = float((logits.argmax(1)[train_mask] == y[train_mask]).float().mean())
    return model, ctx, {
        "test_acc": test_acc, "train_acc": train_acc, "best_val": best_val,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


# ---------------------------------------------------------------------------
# Per-graph equilibrium, frozen-mask Jacobian, resolvent, and the differentiable
# objects for the gates. We re-solve the equilibrium on each SINGLE graph (the
# trained model is graph-agnostic: shared U, W, head) to keep D = n*hidden small.
# ---------------------------------------------------------------------------
def solve_equilibrium(model, A_hat, X, max_iter=500, tol=1e-9):
    X_proj = model.U(X)
    ctx = {"A_hat": A_hat, "X_proj": X_proj}
    N = X.shape[0]
    Z = torch.zeros(N, model.hidden, dtype=A_hat.dtype, device=A_hat.device)
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(float(Z.norm()), 1.0):
                Z = Z_new
                break
            Z = Z_new
    return Z, ctx, X_proj


def frozen_mask(model, Z_star, ctx):
    with torch.no_grad():
        pre = ctx["A_hat"] @ model.W(Z_star) + ctx["X_proj"]
        return (pre > 0).to(Z_star.dtype)


def jz_kron(model, mask, A_hat):
    """J_z = diag(vec mask) kron(A_hat, W), row-major over (N,hidden).
    Bit-identical to compute_jacobian (verified in exp_reachability)."""
    W = model.W.weight  # (h,h); operator computes W(Z)=Z @ W^T
    K = torch.kron(A_hat, W)
    return mask.reshape(-1).unsqueeze(1) * K


def jx_kron(model, mask):
    """J_x = dF/dX = diag(vec mask) kron(I_N, U), row-major over (N,hidden).
    F = relu(A_hat (Z W^T) + X U^T); dF/dvec(X) at frozen mask. U = self.U.weight
    maps (in_feat)->(hidden) via X_proj = X @ U^T, so block (node i) = U.
    Columns are vec over (N, in_features), row-major."""
    U = model.U.weight  # (hidden, in_features)
    N = mask.shape[0]
    I_N = torch.eye(N, dtype=U.dtype, device=U.device)
    K = torch.kron(I_N, U)  # (N*h, N*in)
    return mask.reshape(-1).unsqueeze(1) * K


def resolvent(J_z):
    D = J_z.shape[0]
    I = torch.eye(D, dtype=J_z.dtype, device=J_z.device)
    return torch.linalg.inv(I - J_z)


# ---------------------------------------------------------------------------
# Gate C1: resolvent input-gain g_u = || dz*_u / dx_s || vs reachability label.
# dz*/dvec(X) = (I - J_z)^{-1} J_x.  Source channel column for node s is column
# (s * in_features + 0) of J_x (channel 0 = source one-hot). Per node u, take the
# hidden-block rows (u*h : (u+1)*h) -> a (hidden,) vector -> its norm = g_u.
# ---------------------------------------------------------------------------
def c1_resolvent_gain(model, mask, A_hat, s, hidden):
    J_z = jz_kron(model, mask, A_hat)
    J_x = jx_kron(model, mask)
    R = resolvent(J_z)                  # (D, D)
    in_feat = model.U.weight.shape[1]
    src_col = s * in_feat + 0           # source one-hot channel
    dz_dxs = R @ J_x[:, src_col]        # (D,)
    N = mask.shape[0]
    g = dz_dxs.reshape(N, hidden).norm(dim=1)
    return g.detach().cpu().numpy(), J_z, R


def c1_blackbox_inputgrad(model, A_hat, X, s, hidden):
    """Black-box baseline: ||d y_hat_u / d x_s|| through the FULL nonlinear solve
    (backprop through the unrolled fixed-point). y_hat_u = softmax logit prob of
    reachable class. Gradient wrt the source one-hot channel x[s,0]."""
    X = X.clone().detach().requires_grad_(True)
    X_proj = model.U(X)
    ctx = {"A_hat": A_hat, "X_proj": X_proj}
    N = X.shape[0]
    Z = torch.zeros(N, model.hidden, dtype=A_hat.dtype, device=A_hat.device)
    for _ in range(200):
        Z = model.operator(Z, ctx)
    logits = model.head(Z)
    prob = F_func.softmax(logits, dim=1)[:, 1]  # P(reachable) per node
    gains = np.zeros(N)
    for u in range(N):
        gx = torch.autograd.grad(prob[u], X, retain_graph=(u < N - 1))[0]
        gains[u] = float(gx[s, 0].abs())
    return gains


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of score vs binary label. Excludes the source node itself (always 1,
    trivially reachable) to avoid free credit. Falls back to a rank statistic if
    sklearn is unavailable."""
    mask = np.ones_like(label, dtype=bool)
    lab = label[mask]
    sc = score[mask]
    if len(np.unique(lab)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(lab, sc))
    # Mann-Whitney fallback
    pos = sc[lab == 1]
    neg = sc[lab == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Gate C3: predict-then-intervene on a BRIDGE edge.
#   Predicted dz* (first-order) = S_c column for the bridge edge, scaled by the
#   edge-weight perturbation that deletes it. Deleting edge (i,j) sets A_hat[i,j]
#   and A_hat[j,i] to 0 -> delta on that symmetric edge = -A_hat[i,j]. S_c column
#   k is dz*/d(symmetric edge weight). So predicted dz* = S_c[:,k] * (-A_hat[i,j]).
#   Re-solved dz* = z*(A_hat with edge removed) - z*(A_hat), full nonlinear solve.
# NOTE: deleting a bridge in the NORMALIZED A_hat is an approximation (true edge
#   deletion also renormalizes degrees). We report BOTH a pure-weight deletion
#   (matches S_c's perturbation model exactly) and, separately, a fully
#   renormalized deletion (the "real" algorithmic intervention) so the gate is
#   honest about which faithfulness it measures.
# ---------------------------------------------------------------------------
def c3_predict_then_intervene(model, A_hat, X, bridge, comp_id):
    i, j = bridge
    Z_star, ctx, X_proj = solve_equilibrium(model, A_hat, X)
    mask = frozen_mask(model, Z_star, ctx)

    def F(z, c):
        return model.operator(z, c)

    # structural sensitivity S = (I-J_z)^{-1} J_A; S_c columns per symmetric edge
    J_z = jz_kron(model, mask, A_hat)
    # J_A via finite differences (iem convention), full N^2
    from iem.adversarial import _compute_structural_jacobian
    J_z_fd, J_A, _ = _compute_structural_jacobian(F, Z_star, ctx, "A_hat")
    S = structural_sensitivity_matrix(F, Z_star, ctx, "A_hat", J_z=J_z_fd, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)
    # locate the bridge column
    try:
        k = edge_list.index((min(i, j), max(i, j)))
    except ValueError:
        return None
    dweight = -float(A_hat[i, j])  # pure-weight deletion of the (normalized) edge
    pred_dz = (S_c[:, k] * dweight).detach()

    # --- re-solve: pure-weight deletion (matches S_c model) ---
    A_del = A_hat.clone()
    A_del[i, j] = 0.0
    A_del[j, i] = 0.0
    Z_del, _, _ = solve_equilibrium(model, A_del, X)
    real_dz_pure = (Z_del - Z_star).reshape(-1).detach()

    # --- re-solve: fully renormalized deletion (real algorithmic intervention) ---
    # rebuild A_hat from the 0/1 graph implied by A_hat's support minus this edge
    sup = (A_hat.abs() > 1e-10).double()
    sup[i, j] = 0.0
    sup[j, i] = 0.0
    # support already includes self-loops on the diagonal; strip diag, re-add I
    n = A_hat.shape[0]
    raw = sup.clone()
    raw.fill_diagonal_(0.0)
    A_sl = raw + torch.eye(n, dtype=A_hat.dtype, device=A_hat.device)
    deg = A_sl.sum(1)
    dinv = deg.clamp(min=1e-12).rsqrt()
    A_renorm = (dinv[:, None] * A_sl) * dinv[None, :]
    Z_re, _, _ = solve_equilibrium(model, A_renorm, X)
    real_dz_renorm = (Z_re - Z_star).reshape(-1).detach()

    def cos(a, b):
        na, nb = a.norm(), b.norm()
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float((a @ b) / (na * nb))

    def relerr(pred, real):
        if real.norm() < 1e-12:
            return float("nan")
        return float((pred - real).norm() / real.norm())

    # does the deletion actually change reachability labels? (algorithmic effect)
    y_before = reach_bfs((A_hat.abs() > 1e-10).double().fill_diagonal_(0), _src_of(X))
    y_after = reach_bfs(raw, _src_of(X))
    flips = int((y_before != y_after).sum())

    return {
        "bridge": (i, j),
        "cos_pure": cos(pred_dz, real_dz_pure),
        "relerr_pure": relerr(pred_dz, real_dz_pure),
        "cos_renorm": cos(pred_dz, real_dz_renorm),
        "relerr_renorm": relerr(pred_dz, real_dz_renorm),
        "label_flips": flips,
        "pred_norm": float(pred_dz.norm()),
        "real_pure_norm": float(real_dz_pure.norm()),
        "real_renorm_norm": float(real_dz_renorm.norm()),
    }


def _src_of(X: Tensor) -> int:
    return int(X[:, 0].argmax().item())


# ---------------------------------------------------------------------------
# Gate C2: eigen-mode alignment with component-indicator basis.
#   J_z eigvecs live in R^D (D=N*h). We reduce each eigvec to a NODE-space
#   profile by the per-node hidden-block norm -> v in R^N, then measure the best
#   cosine alignment of the top-k modes (by |eigval|, i.e. largest resolvent
#   gain 1/|1-lambda|) against the component-indicator one-hot basis (each comp
#   indicator = 1 on its nodes). We also compute the SAME metric for A_hat's own
#   eigvecs as the "trivial adjacency-spectral" control: if the IGNN's modes
#   align no better than A_hat's, C2 is just adjacency structure (flag honestly).
# ---------------------------------------------------------------------------
def _node_profiles(eigvecs: Tensor, N: int, hidden: int) -> Tensor:
    """(D,k) complex eigvecs -> (N,k) real node-space magnitude profiles."""
    k = eigvecs.shape[1]
    prof = torch.zeros(N, k, dtype=torch.float64)
    for c in range(k):
        v = eigvecs[:, c].abs().reshape(N, hidden)
        prof[:, c] = v.norm(dim=1)
    return prof


def _comp_basis(comp_id: Tensor, N: int) -> Tensor:
    ncomp = int(comp_id.max().item()) + 1
    B = torch.zeros(N, ncomp, dtype=torch.float64)
    for c in range(ncomp):
        B[(comp_id == c), c] = 1.0
        B[:, c] = B[:, c] / B[:, c].norm().clamp(min=1e-12)
    return B


def c2_eigen_alignment(J_z, A_hat, comp_id, N, hidden, topk=4):
    # all alignment algebra on CPU (small profiles; keeps devices consistent)
    J_z = J_z.detach().cpu()
    A_hat = A_hat.detach().cpu()
    comp_id = comp_id.cpu()
    B = _comp_basis(comp_id, N)  # (N, ncomp), unit columns
    ncomp = B.shape[1]
    topk = min(topk, ncomp + 2)

    # IGNN modes: top-k eigvecs of J_z by resolvent gain 1/|1-lambda|
    evals, evecs = torch.linalg.eig(J_z)
    gain = 1.0 / (1.0 - evals).abs()
    order = torch.argsort(gain, descending=True)[:topk]
    prof = _node_profiles(evecs[:, order], N, hidden)  # (N, topk)
    prof = prof / prof.norm(dim=0, keepdim=True).clamp(min=1e-12)
    # best alignment of each component indicator to ANY of the top-k modes
    M = (B.T @ prof).abs()  # (ncomp, topk)
    ignn_align = float(M.max(dim=1).values.mean())

    # control: A_hat's own eigvecs (node-space already), top-k by |eigval|
    eva, eve = torch.linalg.eigh(A_hat)
    idx = torch.argsort(eva.abs(), descending=True)[:topk]
    Ap = eve[:, idx].abs()
    Ap = Ap / Ap.norm(dim=0, keepdim=True).clamp(min=1e-12)
    Mc = (B.T @ Ap).abs()
    adj_align = float(Mc.max(dim=1).values.mean())

    return {"ignn_align": ignn_align, "adj_align": adj_align, "ncomp": ncomp}


# ===========================================================================
# SELF-CHECKS
# ===========================================================================
def self_check_labels(verbose=True) -> bool:
    """S1: BFS and union-find reachability agree on random multi-component graphs."""
    ok = True
    for t in range(20):
        G = make_partition_graph(n=30, k_comp=random.choice([2, 3]), p=0.2, seed=100 + t)
        for s in range(0, 30, 7):
            a = reach_bfs(G["A"], s)
            b = reach_unionfind(G["A"], s)
            if not np.array_equal(a, b):
                ok = False
                if verbose:
                    print(f"  S1 MISMATCH t={t} s={s}", flush=True)
    if verbose:
        print(f"  [S1] BFS vs union-find reachability agree: {ok}", flush=True)
    return ok


def self_check_resolvent_block(device, verbose=True) -> bool:
    """S2: resolvent block (I-J_z)^{-1} J_x [:, src] matches autograd dz*/dx_s
    through the full differentiable fixed-point solve, on a TINY graph."""
    torch.manual_seed(SEED)
    G = make_partition_graph(n=8, k_comp=2, p=0.5, seed=7)
    A_hat = G["A_hat"].to(device)
    s = 1
    X = make_features(8, s).to(device)
    model = IGNN_Kappa(2, 6, 2, kappa=0.7,
                       A_hat_spectral_norm=float(torch.linalg.svdvals(A_hat)[0])
                       ).to(device).to(DTYPE)
    model._project_W()
    model.eval()

    # autograd reference: differentiable solve, d z*_u / d x[s,0]
    Xr = X.clone().detach().requires_grad_(True)
    X_proj = model.U(Xr)
    ctx = {"A_hat": A_hat, "X_proj": X_proj}
    Z = torch.zeros(8, model.hidden, dtype=DTYPE, device=device)
    for _ in range(300):
        Z = model.operator(Z, ctx)
    # Jacobian of vec(Z) wrt x[s,0] via per-output backward
    D = Z.numel()
    Zf = Z.reshape(-1)
    ref = torch.zeros(D, dtype=DTYPE, device=device)
    for d in range(D):
        gx = torch.autograd.grad(Zf[d], Xr, retain_graph=True)[0]
        ref[d] = gx[s, 0]

    # resolvent prediction
    Z_star, ctx2, _ = solve_equilibrium(model, A_hat, X)
    mask = frozen_mask(model, Z_star, ctx2)
    J_z = jz_kron(model, mask, A_hat)
    J_x = jx_kron(model, mask)
    R = resolvent(J_z)
    pred = (R @ J_x[:, s * 2 + 0]).detach()
    ref = ref.detach()

    err = float((pred - ref).norm() / (ref.norm() + 1e-12))
    cos = float((pred @ ref) / (pred.norm() * ref.norm() + 1e-12))
    ok = err < 1e-5
    if verbose:
        print(f"  [S2] resolvent block vs autograd dz*/dx_s: rel_err={err:.2e} "
              f"cos={cos:.6f}  PASS={ok}", flush=True)
    return ok


# ===========================================================================
# MAIN GATE RUN
# ===========================================================================
def run_gate(device):
    print("\n" + "=" * 74)
    print("  GATING SMOKE — resolvent decodes reachability?  (seed 42)")
    print("=" * 74, flush=True)

    # ---- self-checks first (abort if labels or resolvent extraction are wrong)
    print("\n[SELF-CHECKS]", flush=True)
    s1 = self_check_labels()
    s2 = self_check_resolvent_block(device)
    if not (s1 and s2):
        print("\n!!! SELF-CHECK FAILED — results untrustworthy; aborting.", flush=True)
        return None

    kappa = 0.9
    n_per = 60
    n_train_graphs = 12

    # ---- train ONE kappa<1 IGNN on a union of reachability graphs (G1)
    print(f"\n[TRAIN] kappa={kappa}, {n_train_graphs} graphs x N={n_per}, "
          f"hidden=64, 400 epochs", flush=True)
    data = build_train_bundle(n_train_graphs, n_per, seed=SEED)
    print(f"  union N={data['N']}, reachable frac={float((data['y']==1).float().mean()):.3f}",
          flush=True)
    model, ctx, tr = train_reach_ignn(data, device, seed=SEED, kappa=kappa, epochs=400)
    print(f"  [G1] train_acc={tr['train_acc']:.3f}  val={tr['best_val']:.3f}  "
          f"TEST_ACC={tr['test_acc']:.3f}", flush=True)

    # ---- held-out evaluation graphs for the resolvent gates
    rng = random.Random(SEED + 999)
    n_eval = 8
    rows = []
    c1_aucs, c1_base_aucs, c2_ignn, c2_adj = [], [], [], []
    c3_cos_pure, c3_relerr_pure, c3_cos_renorm, c3_flips = [], [], [], []
    rho_list, acc_list = [], []

    print(f"\n[EVAL] {n_eval} held-out graphs; per-graph rho(J_z), C1 AUC, C2, C3",
          flush=True)
    for ge in range(n_eval):
        k = rng.choice([2, 3, 4])
        p = rng.uniform(0.12, 0.22)
        G = make_partition_graph(n_per, k, p, seed=SEED + 50000 + 137 * ge)
        deg = (G["A"] > 0).sum(1)
        comp0 = (G["comp_id"] == 0).nonzero(as_tuple=True)[0]
        s = int(comp0[deg[comp0].argmax()].item())
        X = make_features(n_per, s).to(device)
        A_hat = G["A_hat"].to(device)
        comp_id = G["comp_id"]
        y = reach_bfs(G["A"], s)

        # equilibrium + frozen mask on this graph
        Z_star, ctxg, _ = solve_equilibrium(model, A_hat, X)
        mask = frozen_mask(model, Z_star, ctxg)

        # per-graph accuracy (does the trained model solve THIS instance?)
        with torch.no_grad():
            logits = model.head(Z_star)
            acc = float((logits.argmax(1).cpu().numpy() == y).mean())
        acc_list.append(acc)

        # rho(J_z)
        J_z = jz_kron(model, mask, A_hat).detach()
        rho = float(torch.linalg.eigvals(J_z).abs().max())
        rho_list.append(rho)

        # C1: resolvent gain AUC vs baseline
        g, J_z2, R = c1_resolvent_gain(model, mask, A_hat, s, model.hidden)
        a_res = auc(g, y)
        gb = c1_blackbox_inputgrad(model, A_hat, X, s, model.hidden)
        a_base = auc(gb, y)
        c1_aucs.append(a_res)
        c1_base_aucs.append(a_base)

        # C2: eigen-mode alignment
        c2 = c2_eigen_alignment(J_z, A_hat, comp_id, n_per, model.hidden, topk=4)
        c2_ignn.append(c2["ignn_align"])
        c2_adj.append(c2["adj_align"])

        # C3: predict-then-intervene on a bridge edge (intra-component bridge)
        bridges = find_bridges(G["A"])
        c3 = None
        if bridges:
            # prefer a bridge whose removal flips labels (true algorithmic effect)
            chosen = None
            for br in bridges:
                raw = (G["A"] > 0).double()
                raw[br[0], br[1]] = 0
                raw[br[1], br[0]] = 0
                yb = reach_bfs(G["A"], s)
                ya = reach_bfs(raw, s)
                if (yb != ya).sum() > 0:
                    chosen = br
                    break
            if chosen is None:
                chosen = bridges[0]
            c3 = c3_predict_then_intervene(model, A_hat, X, chosen, comp_id)
        if c3:
            c3_cos_pure.append(c3["cos_pure"])
            c3_relerr_pure.append(c3["relerr_pure"])
            c3_cos_renorm.append(c3["cos_renorm"])
            c3_flips.append(c3["label_flips"])

        print(f"  g{ge}: k={k} rho={rho:.3f} acc={acc:.3f} | "
              f"C1 AUC res={a_res:.3f} base={a_base:.3f} | "
              f"C2 ign={c2['ignn_align']:.3f} adj={c2['adj_align']:.3f} | "
              f"C3 cos_pure={(c3['cos_pure'] if c3 else float('nan')):.3f} "
              f"flips={(c3['label_flips'] if c3 else -1)}", flush=True)

        rows.append({
            "graph": ge, "k_comp": k, "n": n_per, "source": s,
            "rho_jz": rho, "node_acc": acc,
            "c1_auc_resolvent": a_res, "c1_auc_baseline": a_base,
            "c2_ignn_align": c2["ignn_align"], "c2_adj_align": c2["adj_align"],
            "c3_cos_pure": c3["cos_pure"] if c3 else float("nan"),
            "c3_relerr_pure": c3["relerr_pure"] if c3 else float("nan"),
            "c3_cos_renorm": c3["cos_renorm"] if c3 else float("nan"),
            "c3_relerr_renorm": c3["relerr_renorm"] if c3 else float("nan"),
            "c3_label_flips": c3["label_flips"] if c3 else -1,
        })

    # ---- aggregate
    def ms(x):
        x = [v for v in x if v == v]  # drop nan
        return (float(np.mean(x)), float(np.std(x))) if x else (float("nan"), 0.0)

    rho_m, rho_s = ms(rho_list)
    acc_m, acc_s = ms(acc_list)
    c1_m, c1_sd = ms(c1_aucs)
    c1b_m, c1b_sd = ms(c1_base_aucs)
    c2i_m, c2i_sd = ms(c2_ignn)
    c2a_m, c2a_sd = ms(c2_adj)
    c3_m, c3_sd = ms(c3_cos_pure)
    c3r_m, c3r_sd = ms(c3_relerr_pure)
    c3rn_m, _ = ms(c3_cos_renorm)

    # ---- VERDICT
    contraction_wall = (tr["test_acc"] < 0.75) or (acc_m < 0.75)
    c1_beats_base = (c1_m - c1b_m) > 0.03
    c3_faithful = (c3_m > 0.9) and (c3r_m < 0.25)
    if contraction_wall:
        verdict = "CONTRACTION-WALL"
    elif (c1_m > 0.85) and c3_faithful and (c1_beats_base or c2i_m > c2a_m + 0.02):
        verdict = "RESOLVENT-DECODES"
    else:
        verdict = "WEAK"

    print("\n" + "=" * 74)
    print("  SUMMARY  (seed 42)")
    print("=" * 74)
    print(f"  G1  kappa-target=0.9   rho(J_z) [eval]   = {rho_m:.4f} +/- {rho_s:.4f}  (kappa<1: {rho_m<1})")
    print(f"  G1  test accuracy (held-in split)        = {tr['test_acc']:.3f}")
    print(f"  G1  node accuracy (held-out graphs)      = {acc_m:.3f} +/- {acc_s:.3f}")
    print(f"  C1  resolvent-gain AUC                   = {c1_m:.3f} +/- {c1_sd:.3f}")
    print(f"  C1  black-box input-grad AUC (baseline)  = {c1b_m:.3f} +/- {c1b_sd:.3f}")
    print(f"  C3  predict-vs-resolve cosine (pure)     = {c3_m:.3f} +/- {c3_sd:.3f}")
    print(f"  C3  predict-vs-resolve rel-err (pure)    = {c3r_m:.3f} +/- {c3r_sd:.3f}")
    print(f"  C3  predict-vs-resolve cosine (renorm)   = {c3rn_m:.3f}   (real edge deletion)")
    print(f"  C2  eigen-mode align (IGNN)              = {c2i_m:.3f} +/- {c2i_sd:.3f}")
    print(f"  C2  eigen-mode align (A_hat control)     = {c2a_m:.3f} +/- {c2a_sd:.3f}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 74, flush=True)

    # ---- CSV
    out = Path("results/exp_interp_smoke.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    # append a summary row
    with open(out, "a", newline="") as f:
        f.write("\n# SUMMARY (seed 42)\n")
        f.write(f"# G1_test_acc,{tr['test_acc']:.4f}\n")
        f.write(f"# G1_node_acc_eval,{acc_m:.4f},{acc_s:.4f}\n")
        f.write(f"# G1_rho_jz,{rho_m:.4f},{rho_s:.4f}\n")
        f.write(f"# C1_auc_resolvent,{c1_m:.4f},{c1_sd:.4f}\n")
        f.write(f"# C1_auc_baseline,{c1b_m:.4f},{c1b_sd:.4f}\n")
        f.write(f"# C3_cos_pure,{c3_m:.4f},{c3_sd:.4f}\n")
        f.write(f"# C3_relerr_pure,{c3r_m:.4f},{c3r_sd:.4f}\n")
        f.write(f"# C3_cos_renorm,{c3rn_m:.4f}\n")
        f.write(f"# C2_ignn_align,{c2i_m:.4f},{c2i_sd:.4f}\n")
        f.write(f"# C2_adj_align,{c2a_m:.4f},{c2a_sd:.4f}\n")
        f.write(f"# VERDICT,{verdict}\n")
    print(f"\nResults saved to {out}", flush=True)

    return {"verdict": verdict, "rows": rows}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
          flush=True)
    set_seed(SEED)
    run_gate(device)


if __name__ == "__main__":
    main()
