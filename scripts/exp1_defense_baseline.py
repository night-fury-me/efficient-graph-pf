#!/usr/bin/env python3
"""EXP-1 (review P1-8): matched defense baseline.

Question. Does penalizing sigma_1(S_c), the structural-sensitivity operator,
buy a better certified-robustness / accuracy frontier than the architecture's
generic Lipschitz knob, the spectral cap c (which bounds ||W_eff||_2 and hence
kappa=||J_z||)? This is the "matched spectral-norm baseline" the review (R2 /
Devil's Advocate) asked for, and it de-Cora-specifies the defense by adding a
second citation graph (Citeseer).

Two defenses, both trained from the validated revision-R2 recipe, each swept to
trace a robustness/accuracy frontier:

  A. sc_penalty    loss = CE + lambda * log sigma_1(S_c), fixed cap c=0.9, sweep
                   lambda.  (Ours: penalizes the exact operator that governs the
                   perturbation response.)
  B. lipschitz_cap loss = CE, lambda=0, sweep the cap c in {0.5..0.9}.  (Generic
                   Lipschitz control on W: the matched baseline.)

The c=0.9 / lambda=0 model is the shared anchor for both frontiers (trained once
per seed as `baseline`).

For every trained model we record the same full-graph metrics as the paper's
tab:defense, through the paper's EXACT analysis / attack / certify helpers (so
nothing drifts):
  test accuracy; sigma_1(S_c) (analysis path); kappa=rho(J_z); ||J_z||_2;
  certified fraction (sound rho_v > eps_cert, T3 curvature);
  AEGIS attack damage ||DeltaZ*|| at eps_attack (leading-SVD perturbation).

Expected result (aggregated over the 10 seeds, plotted in the appendix as
certified_fraction vs accuracy): sc_penalty dominates lipschitz_cap, i.e. more
certified robustness per accuracy point, because it targets the operator rather
than the bare weight norm. Framing per outcome is in
`paper/review/ars_review_2026-06-05/15_experiment_scoping.md` (FIX/REFRAME/CONCEDE).

COMPUTE NOTE. Defense A forms a dense N x N delta_A inside aegis_sigma1, so it is
feasible for Cora (N=2708) and Citeseer (N=3327) but OOMs on Pubmed (N=19717).
Defense A is therefore gated to N <= --max-penalty-N (default 6000); Defense B and
all MEASUREMENTS are matrix-free and run on any dataset. So if you add Pubmed it
gets the cap frontier only; the sigma_1 penalty there needs a sparse/subgraph
penalty path (future work).

PROTOCOL (per the per-experiment rule). Implemented to REUSE the paper code; smoke
-test the wiring with --quick (2 seeds, short) before the full sweep; the CSV is
written incrementally and the run RESUMES (re-running skips finished rows). After
the run, aggregate <- printed at the end (mean +/- std over seeds per config).
Then write `exp1_defense_baseline_findings.md`.

Usage:
    .venv/bin/python scripts/exp1_defense_baseline.py --quick      # smoke test
    .venv/bin/python scripts/exp1_defense_baseline.py              # full default
    .venv/bin/python scripts/exp1_defense_baseline.py \
        --datasets Cora Citeseer --n-seeds 10 \
        --lambdas 0.001 0.003 0.01 0.03 0.1 --caps 0.5 0.6 0.7 0.8 \
        --epochs 150 --cert-sample 400 --penalty-every 1
Output:
    results/exp1_defense_baseline.csv   (one row per trained model; resumable)
    results/exp1_defense_baseline.log
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Reuse the paper's EXACT training + analysis + attack + certify helpers so the
# numbers are byte-comparable with tab:defense (no reimplementation).
from scripts.exp_aegis_regularized_training import (  # noqa: E402
    train_regularized,
    analysis_sigma1,
    attack_damage,
    certified_fraction,
    test_accuracy,
)
from scripts.revision_R2._common import LOADERS, DOWNLOADERS, SEEDS  # noqa: E402


def load_with_test(name, device):
    """Load a dataset KEEPING the standard test split (`_common.load_dataset`
    drops test_mask; tab:defense accuracy uses the standard 1000-node test set)."""
    try:
        d = LOADERS[name]()
    except FileNotFoundError:
        DOWNLOADERS[name]()
        d = LOADERS[name]()
    X = d["X"].to(device)
    A = d["A_hat"].to(device)
    y = d["y"].to(device)
    train_mask = d["train_mask"].to(device)
    test_mask = (d["test_mask"] if "test_mask" in d else ~d["train_mask"]).to(device)
    return X, A, y, train_mask, test_mask, int(d["n_features"]), int(d["n_classes"])


FIELDS = ["dataset", "defense", "c", "lambda", "seed", "N",
          "acc", "sigma1", "kappa", "jz_opnorm",
          "cert_frac", "n_cert", "n_correct", "attack_dmg", "attack_flips",
          "peak_gb", "train_s"]


def measure(model, X, A, y, test_mask, eps_attack, eps_cert, cert_sample, seed):
    """Full-graph metrics via the paper's exact helpers (matrix-free; safe on any N)."""
    acc = test_accuracy(model, X, A, y, test_mask)
    an = analysis_sigma1(model, X, A)
    dmg, flips = attack_damage(model, X, A, an, eps=eps_attack)
    cfrac, ncert, ncorr = certified_fraction(
        model, X, y, an, eps_cert=eps_cert, cert_sample=cert_sample, seed=seed)
    out = dict(acc=round(acc, 4), sigma1=round(an["sigma1"], 4),
               kappa=round(an["kappa"], 4), jz_opnorm=round(an["jz_opnorm"], 4),
               cert_frac=round(cfrac, 4), n_cert=ncert, n_correct=ncorr,
               attack_dmg=round(dmg, 4), attack_flips=flips)
    del an
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _key(dataset, defense, c, lam, seed):
    # string key matching how csv.DictReader returns the columns, for resume.
    return (str(dataset), str(defense), str(c), str(lam), str(seed))


def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add(_key(r["dataset"], r["defense"], r["c"], r["lambda"], r["seed"]))
    return done


def summarize(csv_path, log):
    """Print mean +/- std over seeds per (dataset, defense, c, lambda)."""
    if not csv_path.exists():
        return
    groups = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            k = (r["dataset"], r["defense"], r["c"], r["lambda"])
            groups.setdefault(k, []).append(r)
    log("\n" + "=" * 88)
    log("AGGREGATE (mean +/- std over seeds)")
    log("=" * 88)
    log(f"{'dataset':>9} {'defense':>13} {'c':>4} {'lambda':>7} {'n':>3} "
        f"{'acc':>13} {'cert_frac':>13} {'attack_dmg':>14} {'sigma1':>12}")

    def ms(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s
    for k in sorted(groups):
        rs = groups[k]
        acc = ms([float(r["acc"]) for r in rs])
        cf = ms([float(r["cert_frac"]) for r in rs])
        dm = ms([float(r["attack_dmg"]) for r in rs])
        s1 = ms([float(r["sigma1"]) for r in rs])
        log(f"{k[0]:>9} {k[1]:>13} {k[2]:>4} {k[3]:>7} {len(rs):>3} "
            f"{acc[0]:>6.3f}+/-{acc[1]:<5.3f} {cf[0]:>6.3f}+/-{cf[1]:<5.3f} "
            f"{dm[0]:>7.3f}+/-{dm[1]:<5.3f} {s1[0]:>6.2f}+/-{s1[1]:<5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer"],
                    help="both defenses run where N <= --max-penalty-N; "
                         "larger graphs (e.g. Pubmed) get the cap frontier only")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="explicit seeds (override --n-seeds and the AEGIS_SEEDS env)")
    ap.add_argument("--defenses", nargs="+", default=["sc_penalty", "lipschitz_cap"],
                    choices=["sc_penalty", "lipschitz_cap"],
                    help="which defense sweep(s) to run; baseline always runs")
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.001, 0.003, 0.01, 0.03, 0.1],
                    help="sc_penalty sweep (log-penalty knee is in [1e-3, 1e-1])")
    ap.add_argument("--caps", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8],
                    help="lipschitz_cap sweep (the base cap 0.9 is the shared baseline)")
    ap.add_argument("--base-c", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--penalty-form", choices=["log", "raw"], default="log")
    ap.add_argument("--penalty-every", type=int, default=1)
    ap.add_argument("--k-neumann", type=int, default=30)
    ap.add_argument("--n-power", type=int, default=4)
    ap.add_argument("--eps-attack", type=float, default=0.10)
    ap.add_argument("--eps-cert", type=float, default=0.05)
    ap.add_argument("--cert-sample", type=int, default=400,
                    help="certify this many correct nodes (0=all; 400 ~ 30s, "
                         "unbiased estimate of the fraction)")
    ap.add_argument("--max-penalty-N", type=int, default=6000,
                    help="skip Defense A (sigma_1 penalty) above this N "
                         "(its dense N x N delta_A OOMs, e.g. Pubmed)")
    ap.add_argument("--out", type=str, default="results/exp1_defense_baseline.csv")
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, 1 lambda, 1 cap, 40 epochs smoke test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # seed selection: --seeds  >  AEGIS_SEEDS env (cluster shard)  >  --n-seeds
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif os.environ.get("AEGIS_SEEDS"):
        seeds = [int(s) for s in os.environ["AEGIS_SEEDS"].replace(",", " ").split()]
    else:
        seeds = SEEDS[: args.n_seeds]
    lambdas = list(args.lambdas)
    caps = list(args.caps)
    epochs = args.epochs
    cert_sample = args.cert_sample
    if args.quick:
        seeds, lambdas, caps, epochs, cert_sample = seeds[:2], [0.01], [0.7], 40, 200

    # output is CWD-relative so the cluster's per-label working dir isolates each
    # shard (run_job.sh cd's into results/cluster/<label>); locally CWD is the repo.
    _op = Path(args.out)
    out_path = _op if _op.is_absolute() else (Path.cwd() / _op)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    logf = open(log_path, "a")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    done = load_done(out_path)
    write_header = not out_path.exists()
    csvf = open(out_path, "a", newline="")
    writer = csv.DictWriter(csvf, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        csvf.flush()

    def silent(*a, **k):
        return None

    def train_one(dargs, seed, lam, c):
        # train_regularized(X,A,y,train_mask,n_features,n_classes, device,seed, lam,epochs,
        #                   k_neumann,n_power,penalty_every, ..., c=, penalty_form=, log=)
        return train_regularized(
            *dargs, device, seed, lam=lam, epochs=epochs,
            k_neumann=args.k_neumann, n_power=args.n_power,
            penalty_every=args.penalty_every, c=c,
            penalty_form=args.penalty_form, log=silent)

    def run_point(dataset, defense, c, lam, seed, N, X, A, y, test_mask, dargs):
        if _key(dataset, defense, c, lam, seed) in done:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tl = time.time()
        model = train_one(dargs, seed, lam=lam, c=c)
        met = measure(model, X, A, y, test_mask, args.eps_attack, args.eps_cert,
                      cert_sample, seed)
        peak_gb = (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                   if torch.cuda.is_available() else 0.0)
        row = dict(dataset=dataset, defense=defense, c=c, seed=seed, N=N,
                   peak_gb=peak_gb, train_s=round(time.time() - tl, 1),
                   **{"lambda": lam}, **met)
        writer.writerow(row)
        csvf.flush()
        done.add(_key(dataset, defense, c, lam, seed))
        log(f"  [{dataset} s{seed}] {defense:>13} c={c} lam={lam}: "
            f"acc={met['acc']} sigma1={met['sigma1']} kappa={met['kappa']} "
            f"cert={met['cert_frac']} dmg={met['attack_dmg']} "
            f"peak={peak_gb}GB ({row['train_s']}s)")
        del model

    t0 = time.time()
    log(f"\n=== EXP-1 defense baseline | {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"datasets={args.datasets} seeds={seeds}")
    log(f"lambdas={lambdas} caps={caps} base_c={args.base_c} epochs={epochs} "
        f"penalty_form={args.penalty_form} cert_sample={cert_sample}")
    log(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    for dataset in args.datasets:
        X, A, y, train_mask, test_mask, nfeat, ncls = load_with_test(dataset, device)
        N = int(X.shape[0])
        dargs = (X, A, y, train_mask, nfeat, ncls)
        penalty_ok = N <= args.max_penalty_N
        log(f"\n--- {dataset}: N={N} feat={nfeat} cls={ncls} | sigma_1-penalty "
            f"{'ENABLED' if penalty_ok else 'SKIPPED (N>max_penalty_N)'} ---")
        for seed in seeds:
            # shared baseline (c=base_c, lambda=0)
            run_point(dataset, "baseline", args.base_c, 0.0, seed, N, X, A, y,
                      test_mask, dargs)
            # Defense A: sigma_1(S_c) penalty sweep at c=base_c
            if "sc_penalty" in args.defenses and penalty_ok:
                for lam in lambdas:
                    run_point(dataset, "sc_penalty", args.base_c, lam, seed, N, X, A, y,
                              test_mask, dargs)
            # Defense B: Lipschitz cap sweep at lambda=0
            if "lipschitz_cap" in args.defenses:
                for cc in caps:
                    run_point(dataset, "lipschitz_cap", cc, 0.0, seed, N, X, A, y,
                              test_mask, dargs)

    csvf.close()
    summarize(out_path, log)
    log(f"\nDONE. CSV={out_path}  LOG={log_path}  wall={time.time() - t0:.1f}s")
    logf.close()


if __name__ == "__main__":
    main()
