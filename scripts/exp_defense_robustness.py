#!/usr/bin/env python3
"""DEFENSE robustness: does penalizing sigma_1(S_c) during training actually buy
ROBUST ACCURACY under the AEGIS (S_c-optimal) attack, and at what clean-accuracy
cost?

AEGIS has three legs: (1) a structural-sensitivity DIAGNOSIS (sigma_1(S_c) is the
worst-case equilibrium shift per unit edge perturbation), (2) a certified RADIUS,
and (3) a DEFENSE -- the lambda * sigma_1(S_c) penalty (scripts/
exp_aegis_regularized_training.py) that provably shrinks sigma_1. The validated
defended-model numbers (Cora, lambda=3e-4, 10 seeds; see
paper/review/regularizer_multiseed_findings.md) are sigma_1 ~ 32.6, acc ~ 0.739,
cert_frac ~ 0.823 versus the undefended sigma_1 ~ 319, acc ~ 0.781, cert ~ 0.403.

A reviewer flagged that leg (3) has NO main-text ROBUST-ACCURACY experiment: we
show the penalty lowers sigma_1 and the *equilibrium-shift damage* ||Z_pert-Z||,
but never that the classifier RETAINS MORE CORRECT TEST PREDICTIONS under attack.
This script provides exactly that, head-to-head, undefended vs defended.

WHAT IT DOES (per seed, per dataset)
  Train TWO full-graph IGNNs with the EXACT audited recipe + penalty machinery
  from exp_aegis_regularized_training.train_regularized (c=0.9 spectral cap,
  dropout 0.5, cosine LR, raw sigma_1 penalty):
      - UNDEFENDED: lambda = 0.0
      - DEFENDED  : lambda = 3e-4   (the validated operating point)
  For each model:
      - sigma_1(S_c) and the leading right singular vector v_1 of S_c via the
        ANALYSIS path (exp_aegis_regularized_training.analysis_sigma1 ->
        exp_fullgraph_attack_table.svd_direction -> ScalableSensitivity.top_k_svd).
        v_1 is an EDGE-SPACE unit vector (length |E|).
      - the S_c-optimal attack delta-Ahat* = eps * sym(reshape(v_1)) at
        eps in {0.01, 0.05, 0.10}, applied with the audited symmetric scatter
        (exp_full_attack_table.apply_perturbation), then RECONVERGE the perturbed
        equilibrium (exp_full_attack_table.reconverge) and RE-CLASSIFY test nodes.
  METRICS per (model, eps):
      - ROBUST ACCURACY: test-node accuracy under delta-Ahat* (the headline).
      - clean accuracy (eps = 0), sigma_1(S_c).
      - GAIN = defended robust acc - undefended robust acc.
  Also reports the clean-accuracy COST (undef clean - def clean).

HEADLINE QUESTION: does the defended model keep MATERIALLY higher accuracy under
the S_c-optimal attack than the undefended model, and is that robustness gain
worth the clean-accuracy it costs? If the penalty does NOT improve robust
accuracy (e.g. the clean loss outweighs the robustness gain), this script SAYS SO
plainly -- it directly informs whether AEGIS substantiates or demotes the defense.

SELF-CHECKS (printed; a failure means the harness is mis-wired, not a finding):
  S1  undefended clean acc  >  defended clean acc      (penalty costs accuracy)
  S2  undefended sigma_1(S_c) > defended sigma_1(S_c)  (penalty shrinks sensitivity)
  S3  the attack DROPS accuracy (robust < clean for the undefended model at the
      largest eps) -- else the attack/reconverge is mis-wired (no-op perturbation)
  S4  v_1 is the S_c LEADING right singular vector, not random: ||v_1||=1, length
      == |E|, and ||S_c v_1|| (matrix-free, via op.matvec) reproduces sigma_1 to
      within a few percent (a random unit edge-vector would give << sigma_1).

REUSE (nothing validated is re-implemented):
  exp_aegis_regularized_training: train_regularized, analysis_sigma1,
      test_accuracy, load_cora   (the audited training loop + analysis path)
  exp_fullgraph_attack_table:     svd_direction (via analysis_sigma1)
  exp_full_attack_table:          apply_perturbation, reconverge
  The ONLY new logic is robust_accuracy(): build delta-Ahat* = eps*v_1, reconverge,
  classify test nodes vs y. (attack_damage in the training script already builds
  the same eps*v_1 perturbation; we add the classify-test-nodes step it lacks.)

RUN: seed-42 smoke ONLY (both models, all three eps, Cora [+ Citeseer if it loads
cleanly]). The full 10-seed sweep is gated behind --full (do NOT run until the
code + smoke are reviewed).

Usage:
    .venv/bin/python scripts/exp_defense_robustness.py \
        [--seed 42] [--epochs 150] [--lam-def 3e-4] [--datasets Cora,Citeseer] \
        [--eps 0.01,0.05,0.10] [--k-neumann 30] [--n-power 4] [--full]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# --- audited training loop + analysis path (full-graph IGNN, c=0.9, raw penalty) ---
from scripts.exp_aegis_regularized_training import (  # noqa: E402
    analysis_sigma1,
    load_cora,
    test_accuracy,
    train_regularized,
)
# --- audited attack scatter + nonlinear reconvergence ---
from scripts.exp_full_attack_table import (  # noqa: E402
    apply_perturbation,
    reconverge,
)

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


# ---------------------------------------------------------------------------
# Optional extra dataset (Citeseer) via the SAME loader plumbing as Cora.
# Returns the Cora-style 7-tuple, or None if the dataset cannot be loaded
# cleanly (kept optional so the smoke never fails on a missing download).
# ---------------------------------------------------------------------------
def load_dataset(name: str, device):
    if name == "Cora":
        return load_cora(device)
    if name == "Citeseer":
        try:
            from iem.examples.ignn_citeseer_pubmed import (
                _download_planetoid,
                _load_planetoid,
            )
            data_dir = PROJ_ROOT / "datasets" / "citeseer"
            try:
                d = _load_planetoid("citeseer", data_dir)
            except FileNotFoundError:
                _download_planetoid("citeseer", data_dir)
                d = _load_planetoid("citeseer", data_dir)
            X = d["X"].to(device); A = d["A_hat"].to(device); y = d["y"].to(device)
            return (X, A, y, d["train_mask"].to(device), d["test_mask"].to(device),
                    int(d["n_features"]), int(d["n_classes"]))
        except Exception as e:  # noqa: BLE001
            print(f"    [skip] Citeseer unavailable ({type(e).__name__}: {e})",
                  flush=True)
            return None
    raise ValueError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------------------
# THE ONLY NEW LOGIC: robust accuracy under the S_c-optimal attack.
# delta-Ahat* = eps * sym(reshape(v_1)); v_1 is the S_c leading right singular
# vector (edge space). Reconverge the perturbed equilibrium, classify test nodes.
# eps=0 returns clean accuracy (no perturbation, same reconverge path).
# ---------------------------------------------------------------------------
@torch.no_grad()
def robust_accuracy(model, A, y, test_mask, an, eps: float) -> float:
    """Test-node accuracy under delta-Ahat* = eps * v_1 (S_c-optimal direction).

    Builds the perturbation with the audited symmetric scatter, reconverges the
    nonlinear equilibrium from the clean Z*, and classifies via the head. At
    eps=0 the scatter is a no-op so this returns the clean accuracy through the
    identical reconverge+head path (so clean and robust acc are strictly
    comparable -- both read off the reconverged Z*, not the training-time logits).
    """
    Z_clean = an["Z_star"]
    ctx = an["ctx"]
    edge_list = an["op"].edge_list
    v1 = an["v1"]
    weights = eps * v1
    A_pert = apply_perturbation(A, edge_list, weights)
    ctx_pert = {**ctx, "A_hat": A_pert}
    Z_pert = reconverge(model, Z_clean, ctx_pert)
    preds = model.head(Z_pert).argmax(dim=1)
    return float((preds[test_mask] == y[test_mask]).float().mean())


# ---------------------------------------------------------------------------
# S4: confirm v_1 really is the S_c leading right singular vector (not random).
# ||S_c v_1|| (matrix-free via op.matvec) must reproduce sigma_1; a random unit
# edge-vector would give a much smaller action. Returns (ok, sc_v1_norm, rand_norm).
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_v1_is_leading(an, tol_rel: float = 0.05):
    op = an["op"]; v1 = an["v1"]; sigma1 = an["sigma1"]
    n_edges = op.num_edges
    ok_shape = (v1.ndim == 1 and v1.shape[0] == n_edges)
    ok_unit = abs(float(v1.norm()) - 1.0) < 1e-4
    sc_v1 = float(op.matvec(v1).norm())                 # ||S_c v_1|| ~ sigma_1
    g = torch.Generator(device=op.device).manual_seed(12345)
    r = torch.randn(n_edges, device=op.device, dtype=op.dtype, generator=g)
    r = r / r.norm()
    sc_rand = float(op.matvec(r).norm())                # ||S_c v_rand|| << sigma_1
    rel = abs(sc_v1 - sigma1) / (abs(sigma1) + 1e-12)
    ok_leading = (rel < tol_rel) and (sc_v1 > sc_rand)
    return {
        "ok": ok_shape and ok_unit and ok_leading,
        "ok_shape": ok_shape, "ok_unit": ok_unit, "ok_leading": ok_leading,
        "n_edges": n_edges, "sc_v1": sc_v1, "sc_rand": sc_rand,
        "sigma1": sigma1, "rel": rel,
    }


# ---------------------------------------------------------------------------
# One (dataset, seed): train both models, attack at every eps, collect rows.
# ---------------------------------------------------------------------------
def run_pair(ds_name, X, A, y, train_mask, test_mask, nfeat, ncls, device,
             seed, lam_def, epochs, eps_list, k_neumann, n_power, log):
    log(f"\n{'='*78}\n[{ds_name}] seed={seed}: training undefended (lam=0) and "
        f"defended (lam={lam_def:g})\n{'='*78}")
    log(f"  N={X.shape[0]} features={nfeat} classes={ncls} "
        f"train={int(train_mask.sum())} test={int(test_mask.sum())}")

    models = {}
    ans = {}
    for tag, lam in (("undef", 0.0), ("def", lam_def)):
        t0 = time.time()
        log(f"\n  --- training {tag} (lambda={lam:g}) ---")
        m = train_regularized(
            X, A, y, train_mask, nfeat, ncls, device, seed, lam=lam,
            epochs=epochs, k_neumann=k_neumann, n_power=n_power,
            penalty_every=1, penalty_form="raw", log=log)  # RAW = the headline penalty
        an = analysis_sigma1(m, X, A)
        models[tag] = m
        ans[tag] = an
        log(f"  {tag}: sigma_1={an['sigma1']:.3f}  kappa={an['kappa']:.4f}  "
            f"||Jz||={an['jz_opnorm']:.4f}  rebuilt={an['rebuilt']}  "
            f"({time.time()-t0:.1f}s)")

    # --- self-check S4 on BOTH models: v_1 is the S_c leading singular vector ---
    v1u = check_v1_is_leading(ans["undef"])
    v1d = check_v1_is_leading(ans["def"])
    log(f"\n  [S4] v_1 is S_c leading singular vector (||S_c v_1|| ~ sigma_1):")
    for tag, c in (("undef", v1u), ("def", v1d)):
        log(f"      {tag}: |E|={c['n_edges']}  ||S_c v1||={c['sc_v1']:.3f}  "
            f"sigma_1={c['sigma1']:.3f}  rel={c['rel']*100:.2f}%  "
            f"||S_c v_rand||={c['sc_rand']:.3f}  -> {'OK' if c['ok'] else 'FAIL'}")

    # --- accuracy at eps=0 (clean) and each attack eps (robust) ---------------
    rows = []
    # clean accuracy via the reconverge+head path (eps=0), AND via the standard
    # forward() for cross-validation (they should match closely).
    clean = {}
    for tag in ("undef", "def"):
        ra0 = robust_accuracy(models[tag], A, y, test_mask, ans[tag], 0.0)
        fwd_acc = test_accuracy(models[tag], X, A, y, test_mask)
        clean[tag] = ra0
        log(f"  clean acc [{tag}]: reconverge-path={ra0:.4f}  forward()={fwd_acc:.4f}")
        rows.append({
            "dataset": ds_name, "seed": seed, "model": tag, "lambda":
            (0.0 if tag == "undef" else lam_def), "eps": 0.0,
            "robust_acc": round(ra0, 4), "clean_acc": round(ra0, 4),
            "sigma1": round(ans[tag]["sigma1"], 4),
            "kappa": round(ans[tag]["kappa"], 4),
        })

    for eps in eps_list:
        ra_u = robust_accuracy(models["undef"], A, y, test_mask, ans["undef"], eps)
        ra_d = robust_accuracy(models["def"], A, y, test_mask, ans["def"], eps)
        for tag, ra in (("undef", ra_u), ("def", ra_d)):
            rows.append({
                "dataset": ds_name, "seed": seed, "model": tag,
                "lambda": (0.0 if tag == "undef" else lam_def), "eps": eps,
                "robust_acc": round(ra, 4), "clean_acc": round(clean[tag], 4),
                "sigma1": round(ans[tag]["sigma1"], 4),
                "kappa": round(ans[tag]["kappa"], 4),
            })

    # --- self-checks S1/S2/S3 -------------------------------------------------
    eps_max = max(eps_list)
    ra_u_max = robust_accuracy(models["undef"], A, y, test_mask, ans["undef"], eps_max)
    s1 = clean["undef"] > clean["def"]                       # undef cleaner
    s2 = ans["undef"]["sigma1"] > ans["def"]["sigma1"]       # undef higher sigma_1
    s3 = ra_u_max < clean["undef"] - 1e-6                    # attack drops acc
    log(f"\n  [self-checks] S1 undef_clean>def_clean: {s1} "
        f"({clean['undef']:.4f} vs {clean['def']:.4f}) | "
        f"S2 undef_sigma1>def_sigma1: {s2} "
        f"({ans['undef']['sigma1']:.2f} vs {ans['def']['sigma1']:.2f}) | "
        f"S3 attack_drops_acc: {s3} "
        f"(undef robust@{eps_max:g}={ra_u_max:.4f} < clean={clean['undef']:.4f})")
    s4_ok = v1u["ok"] and v1d["ok"]

    checks = {"S1": s1, "S2": s2, "S3": s3, "S4": s4_ok}
    # free analysis-operator graphs before the next (dataset, seed)
    del ans, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, checks


# ---------------------------------------------------------------------------
# Pretty per-eps head-to-head table + gain + clean-cost, for ONE (dataset,seed).
# ---------------------------------------------------------------------------
def print_table(ds_name, seed, rows, eps_list, log):
    by = {(r["model"], r["eps"]): r for r in rows}
    clean_u = by[("undef", 0.0)]["clean_acc"]
    clean_d = by[("def", 0.0)]["clean_acc"]
    sig_u = by[("undef", 0.0)]["sigma1"]
    sig_d = by[("def", 0.0)]["sigma1"]
    log(f"\n  RESULTS  [{ds_name}, seed {seed}]")
    log(f"    sigma_1(S_c):  undef={sig_u:.2f}   def={sig_d:.2f}   "
        f"(ratio {sig_u/max(sig_d,1e-9):.1f}x)")
    log(f"    clean acc   :  undef={clean_u:.4f}  def={clean_d:.4f}  "
        f"(clean-acc cost = {clean_u-clean_d:+.4f})")
    hdr = f"    {'eps':>6} | {'undef robust':>12} {'def robust':>11} {'gain(def-undef)':>16}"
    log(hdr)
    log("    " + "-" * (len(hdr) - 4))
    for eps in eps_list:
        ru = by[("undef", eps)]["robust_acc"]
        rd = by[("def", eps)]["robust_acc"]
        log(f"    {eps:>6.2f} | {ru:>12.4f} {rd:>11.4f} {rd-ru:>+16.4f}")
    # verdict for this (dataset, seed)
    gains = [by[("def", e)]["robust_acc"] - by[("undef", e)]["robust_acc"]
             for e in eps_list]
    cost = clean_u - clean_d
    max_gain = max(gains)
    helps = max_gain > cost and max_gain > 0.0
    # a STRONG result: defended robust acc beats undefended at every eps, by more
    # than the clean-acc it cost.
    strong = all(g > 0 for g in gains) and max_gain > cost
    log(f"    -> max robust gain = {max_gain:+.4f}; clean cost = {cost:+.4f}; "
        f"defense {'HELPS' if helps else 'does NOT help'} on net"
        f"{' (STRONG: wins at every eps, gain>cost)' if strong else ''}")
    return {"clean_cost": cost, "gains": gains, "max_gain": max_gain,
            "helps": helps, "strong": strong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam-def", type=float, default=3e-4,
                    help="defended-model penalty weight (validated operating point)")
    ap.add_argument("--datasets", type=str, default="Cora,Citeseer")
    ap.add_argument("--eps", type=str, default="0.01,0.05,0.10")
    ap.add_argument("--k-neumann", type=int, default=30)
    ap.add_argument("--n-power", type=int, default=4)
    ap.add_argument("--full", action="store_true",
                    help="run the 10-seed sweep over all preferred seeds (gated; "
                         "do NOT run until the code + seed-42 smoke are reviewed)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_list = [float(x) for x in args.eps.split(",") if x.strip()]
    ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    seeds = SEEDS if args.full else [args.seed]

    results_dir = PROJ_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    tag = "full" if args.full else f"smoke_s{args.seed}"
    log_path = results_dir / f"defense_robustness_{tag}.log"
    csv_path = results_dir / f"defense_robustness_{tag}.csv"
    logf = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    t0 = time.time()
    log(f"=== DEFENSE robustness: sigma_1(S_c) penalty vs robust accuracy ===")
    log(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"mode={'FULL 10-seed' if args.full else 'SMOKE (seed %d)' % args.seed}  "
        f"datasets={ds_names}  eps={eps_list}  lam_def={args.lam_def:g}  "
        f"epochs={args.epochs}")

    all_rows = []
    all_checks = []        # (ds, seed, checks dict)
    all_verdicts = []      # (ds, seed, verdict dict)
    for ds_name in ds_names:
        for seed in seeds:
            data = load_dataset(ds_name, device)
            if data is None:
                log(f"[{ds_name}] skipped (loader unavailable).")
                break
            X, A, y, train_mask, test_mask, nfeat, ncls = data
            rows, checks = run_pair(
                ds_name, X, A, y, train_mask, test_mask, nfeat, ncls, device,
                seed, args.lam_def, args.epochs, eps_list, args.k_neumann,
                args.n_power, log)
            all_rows.extend(rows)
            all_checks.append((ds_name, seed, checks))
            v = print_table(ds_name, seed, rows, eps_list, log)
            all_verdicts.append((ds_name, seed, v))

    # --- write CSV ------------------------------------------------------------
    if all_rows:
        fieldnames = ["dataset", "seed", "model", "lambda", "eps",
                      "robust_acc", "clean_acc", "sigma1", "kappa"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # --- self-check summary (a FAIL here = mis-wired harness, not a finding) ---
    log("\n" + "=" * 78)
    log("SELF-CHECK SUMMARY (S1 undef cleaner | S2 undef higher sigma_1 | "
        "S3 attack drops acc | S4 v1 is S_c leading SV)")
    log("=" * 78)
    all_ok = True
    for ds_name, seed, c in all_checks:
        ok = all(c.values())
        all_ok = all_ok and ok
        log(f"  [{ds_name} s{seed}] S1={c['S1']} S2={c['S2']} S3={c['S3']} "
            f"S4={c['S4']}  -> {'ALL PASS' if ok else 'CHECK FAILED'}")
    if not all_ok:
        log("  !!! At least one self-check FAILED -- the attack/training harness "
            "may be mis-wired; do NOT trust the robust-accuracy numbers until "
            "resolved.")

    # --- HEADLINE verdict -----------------------------------------------------
    log("\n" + "=" * 78)
    log("HEADLINE: does the sigma_1(S_c) penalty buy robust accuracy?")
    log("=" * 78)
    for ds_name, seed, v in all_verdicts:
        log(f"  [{ds_name} s{seed}] clean-acc cost={v['clean_cost']:+.4f}  "
            f"robust gains (per eps)={['%+.4f' % g for g in v['gains']]}  "
            f"max gain={v['max_gain']:+.4f}  -> "
            f"{'HELPS' if v['helps'] else 'does NOT help'}"
            f"{' (STRONG)' if v['strong'] else ''}")
    # honest synthesis (smoke = single seed; full = aggregate hint)
    any_helps = any(v["helps"] for _, _, v in all_verdicts)
    all_help = all(v["helps"] for _, _, v in all_verdicts) and bool(all_verdicts)
    if all_help:
        log("\n  VERDICT: the defense HELPS -- on every (dataset, seed) here the "
            "defended model's robust-accuracy gain under the S_c-optimal attack "
            "exceeds its clean-accuracy cost. Substantiates AEGIS leg (3).")
    elif any_helps:
        log("\n  VERDICT: MIXED -- the defense helps on some (dataset, seed) but "
            "not all; the net robust-accuracy benefit is marginal relative to the "
            "clean-accuracy cost. Report honestly; do not overclaim leg (3).")
    else:
        log("\n  VERDICT: the defense does NOT improve robust accuracy on net here "
            "-- the clean-accuracy loss outweighs the robustness gain under the "
            "S_c-optimal attack. This DEMOTES the headline robust-accuracy claim "
            "for leg (3): the penalty shrinks sigma_1 and equilibrium-shift damage "
            "(the diagnosis/analysis result still holds), but that does not "
            "translate into more correct test predictions under attack. Report "
            "this plainly. (Single seed in smoke; confirm with --full.)")

    log(f"\nCSV: {csv_path}")
    log(f"LOG: {log_path}")
    log(f"total wall {time.time()-t0:.1f}s")
    logf.close()


if __name__ == "__main__":
    main()
