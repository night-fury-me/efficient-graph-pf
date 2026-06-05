"""EXP-2 (P1-7) — unification-value ablation: one knob, coherent effects + independent-attacker transfer.

Rebuts the Devil's-Advocate "the coupling is definitional" (DA-M1): shows that a
SINGLE training knob -- the sigma_1(S_c) penalty -- coherently moves the whole
triad {AEGIS attack damage, AEGIS-Conformal certified fraction, sigma_1} AND an
INDEPENDENT iterative attacker's damage (faithful GR-BCD), and lets us contrast
the sigma_1(S_c) knob against generic spectral-norm capping on the GR-BCD-damage
-vs-accuracy frontier.

Honest caveat on "independence": the faithful GR-BCD here optimizes the
equilibrium shift ||Delta Z*|| (the metric with signal on these feature-dominated
subgraphs; classification flips do not move -- see EXP-3), and sigma_1(S_c) is the
worst-case ||Delta Z*||, so "penalizing sigma_1 lowers GR-BCD's damage" is partly
by construction. The non-definitional content is (i) the MAGNITUDE and coherence
of the co-movement across independent readings, and (ii) whether the sigma_1(S_c)
knob blunts GR-BCD MORE than a generic ||W|| cap at MATCHED accuracy. The decisive
rebuttal is the companion compute/completeness table (one rSVD query yields the
full triad at 10^2-10^4x below the union of three separate tools); this script
produces the co-movement + frontier data.

Reuses EXP-1's training/measurement (`exp1_defense_baseline.load_with_test`,
`.measure`, `exp_aegis_regularized_training.train_regularized`) and EXP-3's
faithful attacker (`exp3_sota_attack_sweep.grbcd_order`). Same grid as EXP-1
(baseline + sigma_1-penalty sweep + cap sweep, Cora+Citeseer, 10 seeds), with two
added columns: the faithful GR-BCD and AEGIS equilibrium-shift damage at a fixed
budget on a 50-node subgraph.

Cluster contract: AEGIS_SEEDS env shard, CWD-relative output, resume at
(dataset, defense, c, lambda, seed).

Usage:
    .venv/bin/python scripts/exp2_unification_ablation.py
    AEGIS_SEEDS=42 .venv/bin/python scripts/exp2_unification_ablation.py --datasets Cora
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp1_defense_baseline import load_with_test, measure
from scripts.exp_aegis_regularized_training import train_regularized
from scripts.exp3_sota_attack_sweep import (
    aegis_rankings, _edge_tensors, grbcd_order, deletion_damage,
)
from iem.adversarial import extract_ego_subgraph
from scripts.revision_R2._common import SEEDS

LAMBDAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]     # sigma_1(S_c) penalty sweep (EXP-1 grid)
CAPS = [0.5, 0.6, 0.7, 0.8]                  # lipschitz cap sweep (EXP-1 grid)
BASE_C = 0.9
SUBGRAPH_N = 50
GRBCD_BUDGET = 10
GRBCD_EPOCHS = 125
MAX_PENALTY_N = 6000                         # sigma_1 penalty OOMs above this (EXP-1)

FIELDS = [
    "dataset", "defense", "c", "lambda", "seed", "N",
    "acc", "sigma1", "kappa", "cert_frac", "attack_dmg",
    "n_edges_sub", "grbcd_dmg_sub", "aegis_dmg_sub",
]


def _silent(*a, **k):
    return None


def grbcd_transfer(model, X, A, budget=GRBCD_BUDGET):
    """Faithful GR-BCD (||Delta Z*|| objective) + AEGIS damage at top-k on a
    50-node subgraph. Returns (n_edges, grbcd_dmg, aegis_dmg)."""
    idx = extract_ego_subgraph(A, max_nodes=SUBGRAPH_N)
    X_sub, A_sub = X[idx], A[idx][:, idx]
    w_order, _uw, edge_list, Z_clean = aegis_rankings(model, X_sub, A_sub)
    nE = len(edge_list)
    if nE < 3:
        return nE, float("nan"), float("nan")
    ii, jj, avals = _edge_tensors(A_sub, edge_list)
    k = min(budget, nE)
    grbcd_o = grbcd_order(model, X_sub, A_sub, ii, jj, avals, Z_clean, k, GRBCD_EPOCHS)
    grbcd_dmg = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, grbcd_o, k)
    aegis_dmg = deletion_damage(model, X_sub, A_sub, ii, jj, avals, Z_clean, w_order, k)
    return nE, grbcd_dmg, aegis_dmg


def resolve_seeds(args):
    if args.seeds:
        return list(args.seeds)
    env = os.environ.get("AEGIS_SEEDS", "").strip()
    if env:
        return [int(s) for s in env.replace(",", " ").split()]
    return SEEDS[:args.n_seeds]


def load_done(out_path):
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["defense"], r["c"], r["lambda"], int(r["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--defenses", nargs="+", default=["baseline", "sc_penalty", "lipschitz_cap"],
                    choices=["baseline", "sc_penalty", "lipschitz_cap"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--k-neumann", type=int, default=30)
    ap.add_argument("--n-power", type=int, default=4)
    ap.add_argument("--cert-sample", type=int, default=400)
    ap.add_argument("--eps-attack", type=float, default=0.10)
    ap.add_argument("--eps-cert", type=float, default=0.05)
    ap.add_argument("--grbcd-budget", type=int, default=GRBCD_BUDGET)
    ap.add_argument("--out", default="results/exp2/exp2_unification_ablation.csv")
    args = ap.parse_args()

    seeds = resolve_seeds(args)
    _op = Path(args.out)
    out_path = _op if _op.is_absolute() else Path.cwd() / _op
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = load_done(out_path)
    write_header = not out_path.exists()
    csvf = out_path.open("a", newline="")
    writer = csv.DictWriter(csvf, fieldnames=FIELDS)
    if write_header:
        writer.writeheader(); csvf.flush()

    from datetime import datetime
    print(f"=== EXP-2 unification ablation | {datetime.now():%Y-%m-%d %H:%M:%S} ===")
    print(f"datasets={args.datasets} seeds={seeds} defenses={args.defenses} "
          f"grbcd_budget={args.grbcd_budget} "
          f"device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    # (defense, c, lambda) points to train/measure for one (dataset, seed)
    def points(N):
        pts = []
        if "baseline" in args.defenses:
            pts.append(("baseline", BASE_C, 0.0))
        if "sc_penalty" in args.defenses and N <= MAX_PENALTY_N:
            pts += [("sc_penalty", BASE_C, lam) for lam in LAMBDAS]
        if "lipschitz_cap" in args.defenses:
            pts += [("lipschitz_cap", cc, 0.0) for cc in CAPS]
        return pts

    for dataset in args.datasets:
        X, A, y, train_mask, test_mask, nfeat, ncls = load_with_test(dataset, device)
        N = int(X.shape[0])
        print(f"--- {dataset}: N={N} feat={nfeat} cls={ncls} "
              f"(sigma_1 penalty {'on' if N <= MAX_PENALTY_N else 'OFF (N>%d)' % MAX_PENALTY_N}) ---",
              flush=True)
        for seed in seeds:
            for defense, c, lam in points(N):
                key = (dataset, defense, f"{c}", f"{lam}", seed)
                if key in done:
                    continue
                t0 = time.time()
                try:
                    model = train_regularized(
                        X, A, y, train_mask, nfeat, ncls, device, seed,
                        lam=lam, epochs=args.epochs, k_neumann=args.k_neumann,
                        n_power=args.n_power, penalty_every=1, c=c,
                        penalty_form="log", log=_silent)
                    met = measure(model, X, A, y, test_mask, args.eps_attack,
                                  args.eps_cert, args.cert_sample, seed)
                    nE, grbcd_dmg, aegis_dmg = grbcd_transfer(model, X, A, args.grbcd_budget)
                    writer.writerow({
                        "dataset": dataset, "defense": defense, "c": c, "lambda": lam,
                        "seed": seed, "N": N,
                        "acc": met["acc"], "sigma1": met["sigma1"], "kappa": met["kappa"],
                        "cert_frac": met["cert_frac"], "attack_dmg": met["attack_dmg"],
                        "n_edges_sub": nE, "grbcd_dmg_sub": grbcd_dmg, "aegis_dmg_sub": aegis_dmg,
                    })
                    csvf.flush()
                    done.add(key)
                    print(f"  [{dataset} s{seed}] {defense:>13} c={c} lam={lam}: "
                          f"acc={met['acc']:.3f} sig1={met['sigma1']:.1f} "
                          f"cert={met['cert_frac']:.3f} aegis_dmg={met['attack_dmg']:.2f} "
                          f"GR-BCD_sub={grbcd_dmg:.3f} ({time.time()-t0:.1f}s)", flush=True)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:                        # noqa: BLE001
                    print(f"  [err {dataset} s{seed} {defense} c={c} lam={lam}] {exc}", flush=True)

    csvf.close()
    print(f"\nDONE. CSV={out_path}")


if __name__ == "__main__":
    main()
