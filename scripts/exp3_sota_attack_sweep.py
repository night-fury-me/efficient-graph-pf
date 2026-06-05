"""EXP-3 (P1-10) — faithful SOTA structural-attack head-to-head + budget sweep.

Compares AEGIS's one-query edge ranking against the gold-standard gradient
attackers GR-BCD and PR-BCD (Geisler et al. 2021, "Robustness of GNNs at Scale"),
on six datasets across a discrete-budget sweep k in {1,2,5,10,20,50}.

FAITHFULNESS. The attack ALGORITHMS are ported from the official PyG
implementation `torch_geometric/contrib/nn/models/rbcd_attack.py` (the Geisler
authors' own code), verified section by section:
  * GR-BCD (`GRBCDAttack`): greedy gradient block-coordinate -- each step flips
    (here: deletes) the top `budget // epochs` not-yet-flipped edges by gradient.
    The greedy trajectory is nested, so one budget=max_k run yields every k.
  * PR-BCD (`PRBCDAttack`): relax edge weights to [0,1], ascend with the official
    lr schedule `lr = budget/N * base_lr / sqrt(epoch+1)`, project to the budget
    with the official bisection L1-projection (`_project`/`_bisection`), then
    discretize top-k. Run once per k (non-nested).

The official attack takes the loss as a parameter ("to adapt to other tasks you
most likely only need to provide an appropriate loss"). We supply the task-
appropriate loss: the EQUILIBRIUM SHIFT `||Z*(A_pert) - Z*(A_clean)||`, obtained
by autograd straight through the IGNN's fixed-point `forward` (NOT through our
S_c operator) -- so the baseline gradient is independent of AEGIS. On these
feature-dominated subgraphs the classification margin barely moves under any
structural deletion (deleting every edge flips ~3/50 nodes), so the equilibrium
shift -- precisely what AEGIS audits -- is the metric with signal. Both AEGIS and
the iterative attackers are label-free here; the claim is that AEGIS's single
first-order query recovers what the iterative optimizer reaches over many steps.

Threat model: existing-edge deletion only (the same edge support as S_c), so
AEGIS and the attackers rank the identical edge set and Kendall tau is well posed.
On a 50-node subgraph the full existing-edge set is the block, so the official
block-resampling (a scalability device for huge graphs) is unnecessary; we use
the exact full block.

Metrics per (dataset, seed, k): equilibrium-shift damage ||Delta Z*|| of each
attacker's / AEGIS's top-k deletion, and Kendall tau between AEGIS's A_ij*v_ij
ranking and each attacker's edge order.

Cluster contract (shared with exp1): AEGIS_SEEDS env shard, CWD-relative output,
resume at (dataset, seed).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import SEEDS, load_dataset, train_ignn
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)

ALL_DATASETS = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon", "AmazonFraud"]
DEFAULT_K = [1, 2, 5, 10, 20, 50]
FRAUD_DIR = Path("datasets/amazon_fraud")

FIELDS = [
    "dataset", "seed", "k", "n_edges",
    "tau_grbcd", "tau_prbcd", "tau_grbcd_uw", "tau_grbcd_aij",
    "damage_aegis_w", "damage_aegis_uw", "damage_aegis_aij", "damage_grbcd", "damage_prbcd",
    "aegis_w_over_grbcd", "aegis_w_over_prbcd",
]


# ---------------------------------------------------------------------------
def load_any(name):
    if name == "AmazonFraud":
        from iem.examples.ignn_amazon_fraud import _load_amazon_fraud
        d = _load_amazon_fraud(FRAUD_DIR)
        return (d["X"], d["A_hat"], d["y"], d["train_mask"],
                d["n_features"], d["n_classes"])
    return load_dataset(name)


# ---------------------------------------------------------------------------
# AEGIS one-query rankings + the clean equilibrium the attackers/metric reuse
# ---------------------------------------------------------------------------
def aegis_rankings(model, X_sub, A_sub):
    """One S_c build -> (w_order, uw_order, aij_order, edge_list, Z_clean). Orders
    are permutations of range(|E|) giving the 3-way decomposition that answers
    DA-M2 ("is the ranking carried by the edge weight or the sensitivity?"):
      w_order   ranks by A_ij * v_ij  (headline -- both),
      uw_order  ranks by v_ij         (sensitivity only),
      aij_order ranks by A_ij         (edge-weight-only baseline).
    Z_clean is the detached clean equilibrium."""
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
    S = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return None, None, None, [], Z_star.detach()
    v = S_c.norm(dim=0).cpu().numpy()
    a = np.array([float(A_sub[i, j].item()) for (i, j) in edge_list])
    w_order = list(np.argsort(-(a * v)))
    uw_order = list(np.argsort(-v))
    aij_order = list(np.argsort(-a))
    return w_order, uw_order, aij_order, edge_list, Z_star.detach()


# ---------------------------------------------------------------------------
# Equilibrium-shift loss (the attacker MAXIMIZES it), differentiated through the
# IGNN solve -- independent of S_c.
# ---------------------------------------------------------------------------
def _eq_loss(model, X_sub, A, Z_clean):
    _, Z_pert, _ = model(X_sub, A)
    return (Z_pert - Z_clean).norm()


# Existing-edge deletion: delta_A places -w_e * A_ij on (i,j) and (j,i).
def _edge_tensors(A_sub, edge_list):
    dev = A_sub.device
    ii = torch.tensor([i for (i, j) in edge_list], device=dev, dtype=torch.long)
    jj = torch.tensor([j for (i, j) in edge_list], device=dev, dtype=torch.long)
    avals = A_sub[ii, jj].clone()
    return ii, jj, avals


def _delta_A(A_sub, ii, jj, w, avals):
    vals = -w * avals
    dA = torch.zeros_like(A_sub)
    dA = dA.index_put((ii, jj), vals)
    dA = dA.index_put((jj, ii), vals)
    return dA


# ---------------------------------------------------------------------------
# Official PR-BCD budget projection (PyG _project / _bisection)
# ---------------------------------------------------------------------------
def _bisection(values, a, b, n_pert, eps=1e-5, max_iter=1000):
    def shift(off):
        return float(torch.clamp(values - off, 0, 1).sum().item()) - n_pert
    sa = shift(a)
    miu = a
    for _ in range(int(max_iter)):
        miu = (a + b) / 2.0
        sm = shift(miu)
        if abs(sm) <= eps:
            break
        if sa * sm < 0:
            b = miu
        else:
            a, sa = miu, sm
    return miu


def _project(values, budget, eps=1e-7):
    if float(torch.clamp(values, 0, 1).sum().item()) > budget:
        left = float((values - 1).min().item())
        right = float(values.max().item())
        miu = _bisection(values, left, right, budget)
        values = values - miu
    return torch.clamp(values, eps, 1 - eps)


# ---------------------------------------------------------------------------
def prbcd_order(model, X_sub, A_sub, ii, jj, avals, Z_clean, budget, epochs, base_lr):
    """PR-BCD: relaxed edge weights, official lr schedule + bisection projection."""
    nE = avals.numel()
    N = A_sub.shape[0]
    w = torch.full((nE,), min(1.0, budget / nE), device=A_sub.device, dtype=A_sub.dtype)
    for ep in range(epochs):
        w = w.detach().requires_grad_(True)
        dA = _delta_A(A_sub, ii, jj, w, avals)
        loss = _eq_loss(model, X_sub, A_sub + dA, Z_clean)
        (grad,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            lr = (budget / N) * base_lr / np.sqrt(ep + 1)
            w = _project(w + lr * grad, budget)
    return list(np.argsort(-w.detach().cpu().numpy()))


def _step_sizes(budget, epochs):
    s = budget // epochs
    if s > 0:
        steps = [s] * epochs
        for i in range(budget % epochs):
            steps[i] += 1
    else:
        steps = [1] * budget
    return steps


def grbcd_order(model, X_sub, A_sub, ii, jj, avals, Z_clean, budget, epochs, probe_eps=1e-2):
    """GR-BCD: greedy gradient edge-flips; budget=max_k gives all smaller k.

    The gradient is read at a tiny epsilon-probe on the not-yet-deleted edges:
    the equilibrium-shift loss is singular at the clean graph (||0||), so we
    probe at a small uniform deletion so ||Delta Z*|| > 0 and the gradient is
    defined; at the first step this makes the top pick the most first-order-
    sensitive edge, exactly the greedy intent.
    """
    nE = avals.numel()
    w_disc = torch.zeros(nE, device=A_sub.device, dtype=A_sub.dtype)
    order = []
    for step in _step_sizes(budget, epochs):
        probe = w_disc.clone()
        probe[w_disc == 0] = probe_eps
        w = probe.detach().requires_grad_(True)
        dA = _delta_A(A_sub, ii, jj, w, avals)
        loss = _eq_loss(model, X_sub, A_sub + dA, Z_clean)
        (grad,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            g = grad.clone()
            g[w_disc > 0] = float("-inf")
            n_avail = int((w_disc == 0).sum().item())
            if n_avail == 0:
                break
            pick = torch.topk(g, min(step, n_avail)).indices.tolist()
            for p in pick:
                w_disc[p] = 1.0
                order.append(p)
    remaining = [e for e in range(nE) if e not in set(order)]
    return order + remaining


# ---------------------------------------------------------------------------
@torch.no_grad()
def deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, order, k):
    """Equilibrium-shift damage ||Z*(A\\top-k) - Z_clean|| of deleting the top-k."""
    sel = order[:k]
    dA = torch.zeros_like(A_sub)
    if sel:
        idx = torch.tensor(sel, device=A_sub.device, dtype=torch.long)
        si, sj, sv = ii[idx], jj[idx], -avals[idx]
        dA = dA.index_put((si, sj), sv).index_put((sj, si), sv)
    _, Z_pert, _ = model(X_sub, A_sub + dA)
    return float((Z_pert - Z_clean).norm().item())


def ranking_tau(order_a, order_b):
    nE = len(order_a)
    if nE < 3:
        return float("nan")
    ra = np.empty(nE, dtype=float); rb = np.empty(nE, dtype=float)
    for r, e in enumerate(order_a): ra[e] = r
    for r, e in enumerate(order_b): rb[e] = r
    tau, _ = kendalltau(ra, rb)
    return float(tau)


# ---------------------------------------------------------------------------
def resolve_seeds(args):
    env = os.environ.get("AEGIS_SEEDS", "").strip()
    if env:
        return [int(s) for s in env.replace(",", " ").split()]
    if args.seeds:
        return list(args.seeds)
    return SEEDS[:args.n_seeds]


def load_done(out_path):
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--k-list", nargs="+", type=int, default=DEFAULT_K)
    ap.add_argument("--subgraph-n", type=int, default=50)
    ap.add_argument("--prbcd-epochs", type=int, default=125)
    ap.add_argument("--prbcd-lr", type=float, default=1000.0)
    ap.add_argument("--grbcd-epochs", type=int, default=125)
    ap.add_argument("--out", default="results/exp3/exp3_sota_attack_sweep.csv")
    args = ap.parse_args()

    seeds = resolve_seeds(args)
    _op = Path(args.out)
    out_path = _op if _op.is_absolute() else Path.cwd() / _op
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = load_done(out_path)
    k_max = max(args.k_list)

    from datetime import datetime
    print(f"=== EXP-3 faithful SOTA attack sweep | {datetime.now():%Y-%m-%d %H:%M:%S} ===")
    print(f"datasets={args.datasets} seeds={seeds} k={args.k_list} "
          f"subgraph_n={args.subgraph_n} prbcd(epochs={args.prbcd_epochs},lr={args.prbcd_lr}) "
          f"grbcd_epochs={args.grbcd_epochs} "
          f"device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    new_rows = []
    for name in args.datasets:
        try:
            X, A_hat, y, train_mask, n_features, n_classes = load_any(name)
        except Exception as exc:                                # noqa: BLE001
            print(f"--- {name}: LOAD FAILED ({exc}); skip ---", flush=True)
            continue
        X, A_hat, y, train_mask = (X.to(device), A_hat.to(device),
                                   y.to(device), train_mask.to(device))
        print(f"--- {name}: N={A_hat.shape[0]} feat={n_features} cls={n_classes} ---", flush=True)
        for seed in seeds:
            if (name, seed) in done:
                print(f"  {name} seed={seed} done; skip", flush=True)
                continue
            t0 = time.time()
            try:
                torch.manual_seed(seed); np.random.seed(seed)
                model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed)
                idx = extract_ego_subgraph(A_hat, max_nodes=args.subgraph_n)
                X_sub, A_sub = X[idx], A_hat[idx][:, idx]

                w_order, uw_order, aij_order, edge_list, Z_clean = aegis_rankings(model, X_sub, A_sub)
                nE = len(edge_list)
                if nE < 3:
                    print(f"  {name} seed={seed} |E|={nE}<3; skip", flush=True)
                    del model; continue
                ii, jj, avals = _edge_tensors(A_sub, edge_list)

                grbcd_o = grbcd_order(model, X_sub, A_sub, ii, jj, avals, Z_clean,
                                      min(k_max, nE), args.grbcd_epochs)
                tau_g = ranking_tau(w_order, grbcd_o)
                tau_g_uw = ranking_tau(uw_order, grbcd_o)
                tau_g_aij = ranking_tau(aij_order, grbcd_o)

                for k in args.k_list:
                    if k > nE:
                        continue
                    prbcd_o = prbcd_order(model, X_sub, A_sub, ii, jj, avals, Z_clean,
                                          k, args.prbcd_epochs, args.prbcd_lr)
                    tau_p = ranking_tau(w_order, prbcd_o)
                    d_aw = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, w_order, k)
                    d_auw = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, uw_order, k)
                    d_aij = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, aij_order, k)
                    d_g = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, grbcd_o, k)
                    d_p = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, prbcd_o, k)
                    new_rows.append({
                        "dataset": name, "seed": seed, "k": k, "n_edges": nE,
                        "tau_grbcd": tau_g, "tau_prbcd": tau_p, "tau_grbcd_uw": tau_g_uw,
                        "tau_grbcd_aij": tau_g_aij,
                        "damage_aegis_w": d_aw, "damage_aegis_uw": d_auw, "damage_aegis_aij": d_aij,
                        "damage_grbcd": d_g, "damage_prbcd": d_p,
                        "aegis_w_over_grbcd": d_aw / max(d_g, 1e-12),
                        "aegis_w_over_prbcd": d_aw / max(d_p, 1e-12),
                    })
                print(f"  {name} seed={seed:5d} |E|={nE:3d} "
                      f"tau(w,GR-BCD)={tau_g:+.3f} tau(w,PR-BCD@maxk)~ t={time.time()-t0:.1f}s",
                      flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:                            # noqa: BLE001
                print(f"  [err {name} seed={seed}] {exc}", flush=True)

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"\nWrote {len(new_rows)} new rows to {out_path} ({len(done)} (dataset,seed) skipped).")


if __name__ == "__main__":
    main()
