"""AEGIS-Stackelberg: certified, dominant active defense by subspace-portfolio
edge hardening (full-graph, matrix-free).

================================  CONCEPT  =================================
The defender HARDENS a set M of |M|=B undirected edges: the attacker may NOT
perturb those edges. The attacker then optimises a symmetric edge-supported
delta-Ahat supported on the UNMONITORED edges E\\M, with ||delta-Ahat||_F <= eps.
The worst-case FIRST-ORDER residual damage after hardening M is

        sigma_1( S_c[:, E\\M] )                                    (residual)

i.e. the top singular value of the constrained sensitivity matrix S_c with the
monitored COLUMNS removed. Lower residual = better defense.

This is the column-restriction (Stackelberg) model and is DISTINCT from
exp_fullgraph_defense.py, which zeroes edges in A_hat and *rebuilds* the operator
(changing the equilibrium). Here the operator and equilibrium z*(A) are FIXED;
hardening only forbids the attacker from using columns M. Matrix-free: we mask
the monitored columns to zero inside matvec/rmatvec (MaskedOp) and run the same
randomized top_k_svd on the masked operator.

============================  UNIT-BASIS (B1)  ============================
ScalableSensitivity._edges_to_delta_A writes v_k into BOTH (i,j) and (j,i), so a
unit edge-vector e_k maps to a delta_A with ||delta_A||_F = sqrt(2). Therefore
the code's singular values are per-||v||_2, and the PHYSICAL per-||delta-Ahat||_F
singular values are  sigma_phys = sigma_code / sqrt(2). We divide ALL reported
sigma / energies by sqrt(2) (B1-corrected, "unit-basis") and we build the
damage-confirmation attack with edge weights w = (eps/sqrt(2)) * v so that
||delta-Ahat||_F == eps exactly. SQRT2 is applied consistently everywhere.

============================  SELECTION METHODS  =========================
  1. SUBSPACE-PORTFOLIO (proposal): one randomized SVD -> top-r right singular
     vectors V_r (|E| x r) and sigmas; per-edge leading-subspace energy
     e_k = sum_{j<=r} sigma_j^2 (V_r)_{k,j}^2 ; pick top-B by e_k. r in {1,5,10}.
     r=1 should reproduce the v_ij rank-1 score.
  2. V_IJ top-B: existing per-edge score v_ij = ||S_c[:,k]||_2 (rank-1).
  3. CENTRALITY NULLS (answers reviewer C-1): degree, betweenness,
     current-flow-betweenness (networkx). top-B by each.
  4. RANDOM (mean over RANDOM_DRAWS draws).
  5. GREEDY-COVERAGE: greedily add the edge that most reduces the current
     residual sigma_1 (re-evaluate each step). Submodular/optimal reference.

============================  MEASUREMENTS  ==============================
  (1) Residual sigma_1(S_c[:, E\\M]) vs B in {5,10,20,50}, every method, every ds.
  (2) DOMINANCE: does portfolio (best r) beat v_ij, all centrality nulls, random?
  (3) CEILING: as B covers the leading-r subspace, residual -> sigma_{r+1}(S_c);
      check residual >= sigma_{r+1} (clean lower bound) empirically.
  (4) SUBMODULAR vs MODULAR: is top-B-by-energy == greedy? equal => coverage is
      effectively MODULAR and top-B is exactly optimal; greedy strictly better
      => submodular ((1-1/e) story). Report which + measured gap.
  (5) DAMAGE CONFIRMATION: best method at B=20: build the residual worst-case
      attack (leading right singular vector of the masked S_c, symmetric
      edge-supported on E\\M, ||.||_F = eps=0.1), reconverge, confirm actual
      equilibrium damage ||z*(A+delta)-z*(A)|| is reduced vs no-defense and vs
      centrality nulls -> sigma_1 reduction translates to real damage reduction.

================================  SCALING CAVEATS  =======================
  - current-flow-betweenness needs a dense Laplacian pseudoinverse (O(N^2..N^3))
    and is intractable at WikiCS scale (N=11701, |E|=215603). It is CAPPED:
    computed only when |E| <= CFB_MAX_EDGES, else skipped (reported as 'n/a').
  - GREEDY re-evaluates sigma_1 per candidate per step (B * |pool| SVDs). On
    large graphs the candidate pool is restricted to the union of the top
    GREEDY_POOL edges by v_ij and by portfolio-r=10 energy (a superset of every
    method's picks at the budgets tested), and this is stated in the CSV/output.
  - WikiCS top_k_svd over |E|=215k is feasible matrix-free but each matvec is a
    Neumann resolve; we keep SVD_K small and reuse one sketch.

Outputs:
  results/stackelberg_coverage_residual.csv   (residual-vs-B, all methods)
  results/stackelberg_coverage_ceiling.csv    (sigma spectrum + ceiling check)
  results/stackelberg_coverage_submod.csv     (top-B-by-energy vs greedy)
  results/stackelberg_coverage_damage.csv     (B=20 reconverged damage)

Usage:
    .venv/bin/python scripts/exp_stackelberg_coverage.py \
        [--datasets Cora,Citeseer,WikiCS] [--seeds 3] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import SEEDS, load_dataset, train_ignn, reconverge
from scripts.exp_fullgraph_attack_table import rho_rayleigh
from iem.scalable import ScalableSensitivity

SQRT2 = math.sqrt(2.0)

# ----------------------------- knobs --------------------------------------
BUDGETS = [5, 10, 20, 50]
EPS = 0.10                          # damage-confirmation budget ||delta-Ahat||_F
DAMAGE_B = 20                       # budget for the reconverged damage check
SVD_SEED = 0                        # fixed sketch -> deterministic, comparable
RHO_REBUILD_THRESH = 0.98          # deep Neumann if rho >= this (accurate v_1)
NEUMANN_DEEP = 3000

# Greedy uses a CHEAP SVD (only the sigma_1 argmin RANKING over candidates is
# needed, not its value) over a small candidate pool. k=2/power=2 reproduces the
# k=4/power=3 greedy SET selection (verified) at ~0.6x the cost. The chosen
# greedy set is later re-scored with the full (K,P) SVD for the gap comparison.
GREEDY_SVD_K = 2
GREEDY_SVD_POWER = 2

# Per-dataset protocol. Full-graph WikiCS is MEMORY-BOUND: ScalableSensitivity's
# _edges_to_delta_A allocates a dense N x N adjacency every matvec and the
# structural JVP/VJP holds dense N x N autograd buffers. On a 24 GB GPU a single
# masked randomized SVD peaks ~14 GB at k=2/power=2 (11.8 s) and OOMs for the
# k>=4 needed by the r>=5 portfolio. So WikiCS runs a REDUCED feasibility
# protocol: 1 seed, r in {1}, k=3, no greedy, no current-flow-betweenness, fewer
# random draws -- and this limitation is reported, not hidden. Cora/Citeseer
# (E ~ 5k, dense N^2 ~ 30-44 MB) run the FULL protocol.
#   r_list   : subspace-portfolio ranks
#   svd_k    : top singular triplets for selection + ceiling (>= max(r_list)+a few)
#   svd_power: randomized-SVD power iterations
#   do_greedy: run greedy-coverage submodular reference
#   do_cfb   : run current-flow-betweenness centrality null
#   rand_draws, greedy_pool: as named
FULL = dict(r_list=[1, 5, 10], svd_k=16, svd_power=7, do_greedy=True,
            do_cfb=True, rand_draws=5, greedy_pool=40)
REDUCED = dict(r_list=[1], svd_k=3, svd_power=3, do_greedy=False,
               do_cfb=False, rand_draws=3, greedy_pool=0)
# datasets with more than this many undirected edges use the REDUCED protocol
REDUCED_EDGE_THRESH = 50000
CFB_MAX_EDGES = 30000              # hard cap on CFB regardless of protocol

OUT_DIR = Path("results")
RES_CSV = OUT_DIR / "stackelberg_coverage_residual.csv"
CEIL_CSV = OUT_DIR / "stackelberg_coverage_ceiling.csv"
SUBMOD_CSV = OUT_DIR / "stackelberg_coverage_submod.csv"
DMG_CSV = OUT_DIR / "stackelberg_coverage_damage.csv"


# =====================================================================
# Masked (column-restricted) operator: S_c[:, E\M]
# =====================================================================
class MaskedOp:
    """Thin wrapper presenting S_c with the monitored columns M zeroed.

    top_k_svd only uses .matvec / .rmatvec / .D / .num_edges / .device / .dtype,
    so we forward those and apply a diagonal keep-projector P (1 on E\\M, 0 on M):
        masked.matvec(v)  = base.matvec(P v)     -> equals S_c[:, E\\M] (v_{E\\M})
        masked.rmatvec(u) = P * base.rmatvec(u)  -> zeros monitored sensitivities
    Hence sigma_1(masked) == sigma_1(S_c[:, E\\M]) and the leading right singular
    vector is supported on E\\M (monitored entries are exactly 0).
    """

    def __init__(self, base: ScalableSensitivity, keep_mask: torch.Tensor):
        self.base = base
        self.keep = keep_mask.to(device=base.device, dtype=base.dtype)  # 1.0 / 0.0
        self.D = base.D
        self.num_edges = base.num_edges
        self.device = base.device
        self.dtype = base.dtype

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.base.matvec(self.keep * v)

    def rmatvec(self, u: torch.Tensor) -> torch.Tensor:
        return self.keep * self.base.rmatvec(u)

    # reuse the matrix-free randomized SVD verbatim
    top_k_svd = ScalableSensitivity.top_k_svd


def _svd(op, k=16, power=7, seed=SVD_SEED):
    """Deterministic matrix-free randomized SVD. Returns (U, sigma_code, Vh)."""
    n = op.num_edges
    if n == 0:
        z = torch.zeros(0, device=op.device, dtype=op.dtype)
        return None, z, z
    torch.manual_seed(seed)
    # NOTE: do NOT wrap in torch.no_grad(). The matrix-free rmatvec calls
    # _vjp_Jz -> torch.autograd.grad, which builds and frees a fresh local graph
    # per call (create_graph=False, retain_graph=False); under no_grad that
    # inner graph is absent and autograd.grad raises. Each call's graph is
    # independent and GC'd, so memory stays bounded across the many SVDs.
    U, sigma, Vh = op.top_k_svd(k=min(k, n), n_power_iter=power)
    return U, sigma, Vh


def residual_sigma1(base, monitored_idx, k=16, power=7):
    """sigma_1(S_c[:, E\\M]) in UNIT-BASIS (per ||delta-Ahat||_F), matrix-free.

    monitored_idx: 1-D LongTensor of hardened (forbidden) column indices.
    Returns (sigma1_unit, v1_unit_edgevec) where v1 is supported on E\\M.
    """
    n = base.num_edges
    keep = torch.ones(n, device=base.device, dtype=base.dtype)
    if monitored_idx is not None and len(monitored_idx) > 0:
        keep[monitored_idx] = 0.0
    masked = MaskedOp(base, keep)
    _, sigma, Vh = _svd(masked, k=k, power=power)
    if sigma.numel() == 0:
        return 0.0, None
    s1 = float(sigma[0].item()) / SQRT2          # B1 / unit-basis correction
    v1 = Vh[0].detach() * keep                   # enforce support on E\\M exactly
    return s1, v1


# =====================================================================
# Selection methods -> return a 1-D LongTensor of B column indices
# =====================================================================
def select_portfolio(sigma_code, Vh, r, B):
    """Top-B edges by leading-r-subspace energy e_k = sum_{j<=r} s_j^2 V_{k,j}^2.

    Energy scale is invariant to the global sqrt(2) (it cancels in the ranking),
    so we use sigma_code directly for selection; reported sigmas are B1-corrected.
    """
    r = min(r, sigma_code.numel())
    s = sigma_code[:r] ** 2                        # (r,)
    V = Vh[:r, :] ** 2                             # (r, |E|)
    energy = (s[:, None] * V).sum(dim=0)           # (|E|,)
    B = min(B, energy.numel())
    return torch.topk(energy, B).indices


def select_vij(vij, B):
    """Top-B by per-edge column norm v_ij = ||S_c[:,k]||_2 (rank-1 score)."""
    B = min(B, vij.numel())
    return torch.topk(vij, B).indices


def select_topk_score(score: torch.Tensor, B):
    B = min(B, score.numel())
    return torch.topk(score, B).indices


def greedy_coverage(base, pool_idx, B, k=GREEDY_SVD_K, power=GREEDY_SVD_POWER):
    """Greedily add the edge from pool_idx that most reduces residual sigma_1.

    Re-evaluates sigma_1 of the masked operator after each tentative add. Cost:
    O(B * |pool|) randomized SVDs -> pool is capped by caller. Uses a CHEAP SVD
    (only the argmin over candidates is needed); the final greedy SET is later
    re-scored with the SAME full SVD as the portfolio for an apples-to-apples
    submodular-vs-modular gap. Returns the ordered chosen column indices."""
    pool = [int(i) for i in pool_idx.tolist()]
    chosen: list[int] = []
    n = base.num_edges
    for _ in range(min(B, len(pool))):
        best_e, best_s = None, float("inf")
        for e in pool:
            if e in chosen:
                continue
            mon = torch.tensor(chosen + [e], device=base.device, dtype=torch.long)
            s1, _ = residual_sigma1(base, mon, k=k, power=power)
            if s1 < best_s - 1e-12:
                best_s, best_e = s1, e
        if best_e is None:
            break
        chosen.append(best_e)
    return chosen


# =====================================================================
# Centrality nulls (networkx) -> per-EDGE scores in op.edge_list order
# =====================================================================
def edge_centrality_scores(base, want_cfb: bool):
    """Return dict name -> per-edge score tensor (|E|,) in edge_list order.

    Node-centrality (degree, betweenness, CFB) is mapped to an edge score by the
    endpoint sum c(i)+c(j): hardening edges incident to central nodes. Betweenness
    uses edge_betweenness_centrality directly. CFB is node current-flow-betweenness
    (capped). All graphs here are connected components used as-is by networkx.
    """
    import networkx as nx

    edges = base.edge_list
    N = base.N
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    scores: dict[str, torch.Tensor] = {}

    # degree: edge score = deg(i)+deg(j)
    deg = dict(G.degree())
    scores["degree"] = torch.tensor(
        [deg[i] + deg[j] for (i, j) in edges], dtype=base.dtype, device=base.device
    )

    # edge betweenness (the natural per-edge centrality null)
    ebc = nx.edge_betweenness_centrality(G, normalized=True)
    scores["betweenness"] = torch.tensor(
        [ebc.get((i, j), ebc.get((j, i), 0.0)) for (i, j) in edges],
        dtype=base.dtype, device=base.device,
    )

    # current-flow betweenness (node) -> endpoint sum; capped for runtime
    if want_cfb and base.num_edges <= CFB_MAX_EDGES:
        try:
            # operate on the largest connected component (CFB needs connectivity)
            comps = list(nx.connected_components(G))
            Gc = G.subgraph(max(comps, key=len)).copy()
            cfb_node = nx.current_flow_betweenness_centrality(Gc, normalized=True)
            scores["cfb"] = torch.tensor(
                [cfb_node.get(i, 0.0) + cfb_node.get(j, 0.0) for (i, j) in edges],
                dtype=base.dtype, device=base.device,
            )
        except Exception as ex:  # pragma: no cover - defensive
            print(f"    [cfb] failed: {ex}; marking n/a", flush=True)
    return scores


# =====================================================================
# Per-edge v_ij column norms (UNIT-BASIS) -- matrix-free, batched columns
# =====================================================================
def edge_vij_unit(base):
    """v_ij = ||S_c[:,k]||_2 per edge, B1-corrected. Uses base.edge_vulnerability
    (one resolvent per edge). Returned in edge_list order, divided by sqrt(2)."""
    vu = base.edge_vulnerability()  # list[(i,j,norm)] sorted desc (needs grad path)
    order = {(i, j): k for k, (i, j) in enumerate(base.edge_list)}
    out = torch.zeros(base.num_edges, dtype=base.dtype, device=base.device)
    for (i, j, val) in vu:
        out[order[(i, j)]] = val / SQRT2
    return out


# =====================================================================
# Build operator (forward to fixed point, accurate Neumann at high rho)
# =====================================================================
def build_op(model, X, A):
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A)
    op = ScalableSensitivity(F_op, Z_star, ctx)
    rho = rho_rayleigh(op)
    rebuilt = False
    if rho >= RHO_REBUILD_THRESH:
        op = ScalableSensitivity(F_op, Z_star, ctx, neumann_terms=NEUMANN_DEEP)
        rebuilt = True
    return op, Z_star, ctx, rho, rebuilt


# =====================================================================
# Damage confirmation: reconverged ||z*(A+delta)-z*(A)||
# =====================================================================
def apply_perturbation_vec(A, rows, cols, weights):
    A_pert = A.clone()
    A_pert[rows, cols] = A[rows, cols] + weights
    A_pert[cols, rows] = A[cols, rows] + weights
    return A_pert


def confirm_damage(model, base, A_hat, Z_star, ctx, monitored_idx, eps=EPS, K=16, P=7):
    """Build the residual worst-case attack on E\\M (leading right singular vector
    of the masked S_c), scaled so ||delta-Ahat||_F == eps, reconverge, return the
    actual equilibrium damage ||z*(A+delta)-z*(A)||."""
    s1, v1 = residual_sigma1(base, monitored_idx, k=K, power=P)
    if v1 is None or float(v1.norm()) < 1e-12:
        return 0.0, s1
    v1 = v1 / v1.norm()                          # unit edge-vector on E\\M
    # ||delta-Ahat||_F = sqrt(2)*||w||_2 = eps  =>  w = (eps/sqrt(2)) * v1
    w = (eps / SQRT2) * v1
    rows = base._edge_idx[:, 0].contiguous()
    cols = base._edge_idx[:, 1].contiguous()
    A_pert = apply_perturbation_vec(A_hat, rows, cols, w)
    with torch.no_grad():
        Z_pert = reconverge(model, Z_star, {**ctx, "A_hat": A_pert})
        damage = float((Z_pert - Z_star).norm())
    return damage, s1


# =====================================================================
# main
# =====================================================================
def run_dataset(ds_name, n_seeds, device, quick):
    print(f"\n==================== {ds_name} ====================", flush=True)
    X, A_hat, y, train_mask, nf, nc = load_dataset(ds_name)
    X = X.to(device); A_hat = A_hat.to(device); y = y.to(device)
    train_mask = train_mask.to(device)

    res_rows, ceil_rows, sub_rows, dmg_rows = [], [], [], []
    seeds = SEEDS[:n_seeds]

    for si, seed in enumerate(seeds):
        t0 = time.time()
        model = train_ignn(X, A_hat, y, train_mask, nf, nc, device, seed)
        base, Z_star, ctx, rho, rebuilt = build_op(model, X, A_hat)
        E = base.num_edges
        if E == 0:
            continue

        # ---- pick protocol from graph size (now that E is known) ----
        cfg = dict(REDUCED if E > REDUCED_EDGE_THRESH else FULL)
        if quick:                      # smoke-test: shrink everything
            cfg = dict(REDUCED); cfg["r_list"] = [1, 5]; cfg["do_greedy"] = True
            cfg["greedy_pool"] = 20
        if E > REDUCED_EDGE_THRESH and si > 0:
            # large graph: only the first seed (cost), matches the stated caveat
            del model, base; gc.collect()
            if device == "cuda": torch.cuda.empty_cache()
            continue
        r_list = [r for r in cfg["r_list"] if r < min(cfg["svd_k"], E)]
        K = min(cfg["svd_k"], E)
        P = cfg["svd_power"]
        reduced = E > REDUCED_EDGE_THRESH

        # one clean SVD -> spectrum + portfolio energies + ceiling sigmas
        _, sigma_code, Vh = _svd(base, k=K, power=P)
        sigma_unit = (sigma_code / SQRT2).cpu().numpy()
        sigma1_clean = float(sigma_unit[0])

        # per-edge v_ij (unit basis); rank-1 portfolio should reproduce this
        vij = edge_vij_unit(base)

        # centrality nulls
        want_cfb = cfg["do_cfb"] and (E <= CFB_MAX_EDGES) and not quick
        cent = edge_centrality_scores(base, want_cfb=want_cfb)

        # ---- selection-index sets (computed at max budget, sliced per B) ----
        port_idx = {r: select_portfolio(sigma_code, Vh, r, max(BUDGETS)) for r in r_list}
        vij_full = select_vij(vij, max(BUDGETS))

        greedy_order = []
        pool = torch.zeros(0, dtype=torch.long, device=device)
        if cfg["do_greedy"]:
            # candidate pool: union of top-pool by v_ij and by highest-r energy
            pool = torch.unique(torch.cat([
                select_vij(vij, cfg["greedy_pool"]),
                select_portfolio(sigma_code, Vh, max(r_list), cfg["greedy_pool"]),
            ]))
            greedy_order = greedy_coverage(base, pool, max(BUDGETS))

        # ---- residual vs B for every method (FULL-SVD scoring) ----
        def record(method, B, monitored):
            s1, _ = residual_sigma1(base, monitored, k=K, power=P)
            res_rows.append(dict(dataset=ds_name, seed=seed, B=B, method=method,
                                 reduced=int(reduced), residual=s1,
                                 sigma1_clean=sigma1_clean,
                                 reduction=1.0 - s1 / max(sigma1_clean, 1e-12)))
            return s1

        rng = np.random.default_rng(seed)
        for B in BUDGETS:
            for r in r_list:
                record(f"portfolio_r{r}", B, port_idx[r][:B])
            record("vij", B, vij_full[:B])
            for cname, cscore in cent.items():
                record(f"cent_{cname}", B, select_topk_score(cscore, B))
            if not want_cfb:
                res_rows.append(dict(dataset=ds_name, seed=seed, B=B,
                                     method="cent_cfb", reduced=int(reduced),
                                     residual=float("nan"), sigma1_clean=sigma1_clean,
                                     reduction=float("nan")))
            rvals = []
            for _ in range(cfg["rand_draws"]):
                ridx = torch.tensor(rng.choice(E, size=min(B, E), replace=False),
                                    device=device, dtype=torch.long)
                rvals.append(residual_sigma1(base, ridx, k=K, power=P)[0])
            rmean = float(np.mean(rvals))
            res_rows.append(dict(dataset=ds_name, seed=seed, B=B, method="random",
                                 reduced=int(reduced), residual=rmean,
                                 sigma1_clean=sigma1_clean,
                                 reduction=1.0 - rmean / max(sigma1_clean, 1e-12)))
            if cfg["do_greedy"]:
                record("greedy", B, torch.tensor(greedy_order[:B], device=device,
                                                 dtype=torch.long))

        # ---- ceiling: residual(portfolio-r at B) vs sigma_{r+1} ----
        for r in r_list:
            sig_rp1 = float(sigma_unit[r]) if r < len(sigma_unit) else float("nan")
            for B in BUDGETS:
                s1, _ = residual_sigma1(base, port_idx[r][:B], k=K, power=P)
                ceil_rows.append(dict(dataset=ds_name, seed=seed, r=r, B=B,
                                      reduced=int(reduced), residual=s1,
                                      sigma_r_plus_1=sig_rp1,
                                      holds=int(s1 >= sig_rp1 - 1e-9),
                                      gap=s1 - sig_rp1))
        for j, sv in enumerate(sigma_unit):  # full spectrum (r=-1 marker rows)
            ceil_rows.append(dict(dataset=ds_name, seed=seed, r=-1, B=j,
                                  reduced=int(reduced), residual=float("nan"),
                                  sigma_r_plus_1=float(sv), holds=-1, gap=float("nan")))

        # ---- submodular vs modular: top-B-by-energy(r) vs greedy ----
        # Re-score BOTH sets with the SAME full SVD (K,P) so the gap is not SVD noise.
        if cfg["do_greedy"]:
            for r in r_list:
                for B in BUDGETS:
                    s_top, _ = residual_sigma1(base, port_idx[r][:B], k=K, power=P)
                    gmon = torch.tensor(greedy_order[:B], device=device, dtype=torch.long)
                    s_grd, _ = residual_sigma1(base, gmon, k=K, power=P)
                    set_top = set(int(x) for x in port_idx[r][:B].tolist())
                    set_grd = set(greedy_order[:B])
                    sub_rows.append(dict(dataset=ds_name, seed=seed, r=r, B=B,
                                         residual_topB=s_top, residual_greedy=s_grd,
                                         greedy_minus_topB=s_grd - s_top,
                                         set_equal=int(set_top == set_grd),
                                         greedy_pool=int(len(pool))))

        # ---- damage confirmation at B=DAMAGE_B: best method + nulls + none ----
        best_r, best_s = r_list[0], float("inf")
        for r in r_list:
            s1, _ = residual_sigma1(base, port_idx[r][:DAMAGE_B], k=K, power=P)
            if s1 < best_s:
                best_s, best_r = s1, r
        methods_for_damage = {
            "none": None,
            f"portfolio_r{best_r}": port_idx[best_r][:DAMAGE_B],
            "vij": vij_full[:DAMAGE_B],
        }
        if cfg["do_greedy"]:
            methods_for_damage["greedy"] = torch.tensor(
                greedy_order[:DAMAGE_B], device=device, dtype=torch.long)
        for cname, cscore in cent.items():
            methods_for_damage[f"cent_{cname}"] = select_topk_score(cscore, DAMAGE_B)
        for mname, mon in methods_for_damage.items():
            dmg, s1 = confirm_damage(model, base, A_hat, Z_star, ctx, mon, eps=EPS, K=K, P=P)
            dmg_rows.append(dict(dataset=ds_name, seed=seed, B=DAMAGE_B,
                                 method=mname, residual_sigma1=s1, damage=dmg,
                                 best_r=best_r if "portfolio" in mname else -1))

        print(f"  seed={seed:5d}  E={E}  rho={rho:.4f}{' [deep]' if rebuilt else ''}  "
              f"sig1={sigma1_clean:.4f}  best_r={best_r}  proto={'REDUCED' if reduced else 'full'}"
              f"  ({time.time()-t0:.1f}s)", flush=True)
        del model, base
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return res_rows, ceil_rows, sub_rows, dmg_rows


def _write(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path}  ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Cora,Citeseer,WikiCS")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="skip CFB everywhere + smaller for smoke-test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_res, all_ceil, all_sub, all_dmg = [], [], [], []
    for ds in datasets:
        r, c, s, d = run_dataset(ds, args.seeds, device, args.quick)
        all_res += r; all_ceil += c; all_sub += s; all_dmg += d
        # incremental flush so a long WikiCS run never loses Cora/Citeseer
        _write(RES_CSV, all_res)
        _write(CEIL_CSV, all_ceil)
        _write(SUBMOD_CSV, all_sub)
        _write(DMG_CSV, all_dmg)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
