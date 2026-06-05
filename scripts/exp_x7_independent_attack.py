"""Exp X7: AEGIS one-query direction vs genuinely INDEPENDENT attackers.

Reviewer concern E2: "Shift-PGD shares AEGIS's IFT gradients, so beating it only
validates the linearized solve, not a real adversary." This experiment pits AEGIS's
single-query SVD direction (top right singular vector of the target's S_c) against two
attackers that share *no* gradients with S_c:

  1. AEGIS          : 1 query  -- eps * Vh[0] of the target S_c (the method).
  2. Black-box RS   : M queries -- best-of-M random unit directions, REAL equilibrium
                       shift measured per draw; no gradients, no S_c. (genuinely independent)
  3. Transfer       : 0 target-internal queries -- an INDEPENDENT surrogate IGNN (different
                       seed) is trained; its own attack direction is crafted on the surrogate
                       and TRANSFERRED to the target. We give the transfer adversary every
                       advantage: best of {surrogate-SVD (+/-), surrogate Cls-PGD}.

All attacks share the SAME budget eps and the SAME apply_perturbation path, so the
comparison is budget-matched. The metric is the REAL reconverged equilibrium shift
||Z_pert* - Z_clean*|| (the non-circular quantity) plus prediction-flip count. AEGIS is
held to its single one-query direction (no best-of, no sign search) -- this biases the test
AGAINST AEGIS, so any win is conservative.

Win condition for X7: at the paper's anchor budget eps=0.05, AEGIS's single-query direction
matches or beats both independent attackers on real damage -> the One-Query direction claim
goes from "defended" to "proven".

Datasets: Cora, Citeseer, Pubmed, Amazon, WikiCS | 10 seeds | eps=0.05 | M=512
Output: results/x7_independent_attack.csv

Usage:
    .venv/bin/python scripts/exp_x7_independent_attack.py
"""

from __future__ import annotations

import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
# Reuse the *verified* attack machinery from the full-attack-table experiment.
from exp_full_attack_table import (
    apply_perturbation,
    measure_attack,
    pgd_attack,
    reconverge,
    set_seed,
    train_ignn,
)

EPS = 0.05            # paper anchor budget (matches Cov@0.05, smoothing sigma, smallest breach)
M = 512               # black-box random-search query budget (strong independent baseline)
SURR_OFFSET = 10007   # surrogate seed = seed + offset (independent parameters, same arch)
SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass


def load_datasets():
    from iem.examples.ignn_amazon import _load_amazon
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_cora import _load_cora
    from iem.examples.ignn_wikics import _load_wikics
    return [
        ("Cora",     lambda: _load_cora(Path("datasets/cora"))),
        ("Citeseer", lambda: _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("Pubmed",   lambda: _load_planetoid("pubmed", Path("datasets/pubmed"))),
        ("Amazon",   lambda: _load_amazon(Path("datasets/amazon_photo"))),
        ("WikiCS",   lambda: _load_wikics(Path("datasets/wikics"))),
    ]


def compute_view(model, X, A_hat, idx, A_sub):
    """Equilibrium, ctx, S_c and edge_list for `model` on the fixed subgraph (idx, A_sub).

    edge_list comes only from A_sub topology, so it is identical across models -> transfer
    directions live in the same coordinate space as the target's.
    """
    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
    Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    del J_z, J_A, S
    return Z_sub, ctx_sub, S_c, edge_list


def simba_blackbox(target, Z_sub, ctx_sub, edge_list, eps, M, gen, preds_clean):
    """SimBA-style query attacker on the L2 (Frobenius) edge ball, maximising the
    REAL reconverged equilibrium shift. Budget = M model queries. A principled
    (adaptive) replacement for pure random search."""
    A_sub = ctx_sub["A_hat"]
    n = len(edge_list)
    delta = torch.zeros(n, device=A_sub.device)
    step = 1.5 * eps / (n ** 0.5)

    def dmg(dl):
        return measure_attack(target, Z_sub, ctx_sub,
                              apply_perturbation(A_sub, edge_list, dl), preds_clean)

    cur_d, _cur_fl = dmg(delta)
    queries = 1
    best_d, best_fl, best_delta = cur_d, _cur_fl, delta.clone()
    while queries < M:
        order = torch.randperm(n, generator=gen, device=A_sub.device)
        improved = False
        for c in order.tolist():
            if queries >= M:
                break
            for s in (step, -step):
                if queries >= M:
                    break
                trial = delta.clone()
                trial[c] += s
                nrm = trial.norm()
                if nrm > eps:
                    trial = trial * (eps / nrm)
                d, fl = dmg(trial)
                queries += 1
                if d > cur_d:
                    delta, cur_d = trial, d
                    improved = True
                    if d > best_d:
                        best_d, best_fl, best_delta = d, fl, trial.clone()
                    break
        if not improved:
            break  # a full coordinate pass with no accepted move -> converged
    return best_delta, best_d, best_fl


def nes_blackbox(target, Z_sub, ctx_sub, edge_list, eps, M, gen, preds_clean,
                 pop=20, sigma_frac=0.1):
    """NES query attacker: antithetic finite-difference gradient estimate of the
    real equilibrium shift, projected ascent on the L2 ball. Budget = M queries."""
    A_sub = ctx_sub["A_hat"]
    n = len(edge_list)
    sigma = sigma_frac * eps
    step = 2.5 * eps / max(M // pop, 1)

    def dmg(dl):
        return measure_attack(target, Z_sub, ctx_sub,
                              apply_perturbation(A_sub, edge_list, dl), preds_clean)

    delta = torch.zeros(n, device=A_sub.device)
    best_d, best_fl, best_delta = -1.0, 0, delta.clone()
    queries = 0
    while queries + pop <= M:
        grad = torch.zeros(n, device=A_sub.device)
        for _ in range(pop // 2):
            u = torch.randn(n, generator=gen, device=A_sub.device)
            u = u / (u.norm() + 1e-12)
            dp, dm = delta + sigma * u, delta - sigma * u
            d_p, f_p = dmg(dp)
            d_m, f_m = dmg(dm)
            queries += 2
            grad += (d_p - d_m) / (2 * sigma) * u
            for cd, cf, cvec in ((d_p, f_p, dp), (d_m, f_m, dm)):
                if cd > best_d:
                    best_d, best_fl, best_delta = cd, cf, cvec.clone()
        gn = grad.norm()
        if gn > 1e-12:
            delta = delta + step * grad / gn
        nrm = delta.norm()
        if nrm > eps:
            delta = delta * (eps / nrm)
    d, fl = dmg(delta)
    if d > best_d:
        best_d, best_fl, best_delta = d, fl, delta.clone()
    return best_delta, best_d, best_fl


def run_single(ds_name, data, seed, device):
    try:
        X = data["X"].to(device)
        A_hat = data["A_hat"].to(device)
        y = data["y"].to(device)

        set_seed(seed)
        idx = extract_ego_subgraph(A_hat, max_nodes=50)   # model-independent, deterministic
        A_sub = A_hat[idx][:, idx].clone()
        y_sub = y[idx]
        n_nodes = len(y_sub)

        # --- INDEPENDENT SURROGATE: trained FIRST, then freed before the target loads, so
        #     only one model is ever resident (avoids OOM on Pubmed/Amazon). Its transfer
        #     directions are captured as plain edge-space vectors that survive the free. ---
        surr = train_ignn(data, device, seed + SURR_OFFSET)
        Zs, ctxs, S_c_s, edge_list_s = compute_view(surr, X, A_hat, idx, A_sub)
        if not edge_list_s:
            return None
        n_edges = len(edge_list_s)
        _, _, Vhs = torch.linalg.svd(S_c_s, full_matrices=False)
        surr_svd = (EPS * Vhs[0]).detach().clone()                       # transfer-SVD direction
        delta_s = pgd_attack(surr, Zs, ctxs, None, edge_list_s, EPS,     # transfer-Cls-PGD
                             objective="classification", y_sub=y_sub).detach().clone()
        del surr, Zs, ctxs, S_c_s, Vhs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- TARGET ---
        set_seed(seed)
        target = train_ignn(data, device, seed)
        Z_sub, ctx_sub, S_c, edge_list = compute_view(target, X, A_hat, idx, A_sub)
        assert len(edge_list) == n_edges, "edge_list mismatch -> transfer misaligned"
        with torch.no_grad():
            preds_clean = target.head(Z_sub).argmax(dim=1)

        # --- 1. AEGIS: single one-query SVD direction (no best-of, no sign search) ---
        _, _, Vh = torch.linalg.svd(S_c, full_matrices=False)
        A_aegis = apply_perturbation(A_sub, edge_list, EPS * Vh[0])
        dmg_aegis, flips_aegis = measure_attack(target, Z_sub, ctx_sub, A_aegis, preds_clean)
        del S_c, Vh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 2. Independent black-box query attackers at the SAME M-query budget.
        #     Pure random search (RS) is the weak reference the baseline audit flagged
        #     as a strawman; SimBA and NES are principled ADAPTIVE query attackers. Each
        #     gets M queries; we report all three and take the STRONGEST as the
        #     black-box. None uses gradients or S_c -> genuinely independent. ---
        g = torch.Generator(device=A_sub.device).manual_seed(seed * 7919 + 1)
        dmg_rs, flips_rs = -1.0, 0
        for _ in range(M):
            v = torch.randn(n_edges, generator=g, device=A_sub.device)
            v = v / v.norm() * EPS
            d, fl = measure_attack(target, Z_sub, ctx_sub,
                                   apply_perturbation(A_sub, edge_list, v), preds_clean)
            if d > dmg_rs:
                dmg_rs, flips_rs = d, fl
        _, dmg_simba, flips_simba = simba_blackbox(target, Z_sub, ctx_sub, edge_list, EPS, M, g, preds_clean)
        _, dmg_nes, flips_nes = nes_blackbox(target, Z_sub, ctx_sub, edge_list, EPS, M, g, preds_clean)
        bb_kind, dmg_bb, flips_bb = max(
            (("rs", dmg_rs, flips_rs), ("simba", dmg_simba, flips_simba), ("nes", dmg_nes, flips_nes)),
            key=lambda t: t[1])

        # --- 3. Transfer: surrogate directions applied to the TARGET (best of svd +/-, Cls-PGD).
        #     SVD is an unsigned direction so both signs are fair game; the PGD delta is already
        #     a signed optimum. We hand the transfer adversary the best of the three. ---
        dmg_tr, flips_tr, which_tr = -1.0, 0, ""
        for sign, vec, kind in ((1.0, surr_svd, "svd+"), (-1.0, surr_svd, "svd-"),
                                (1.0, delta_s, "clspgd")):
            A_t = apply_perturbation(A_sub, edge_list, sign * vec)
            d, fl = measure_attack(target, Z_sub, ctx_sub, A_t, preds_clean)
            if d > dmg_tr:
                dmg_tr, flips_tr, which_tr = d, fl, kind

        return {
            "dataset": ds_name, "seed": seed, "eps": EPS, "M": M,
            "n_nodes": n_nodes, "n_edges": n_edges,
            "dmg_aegis": dmg_aegis, "dmg_blackbox": dmg_bb, "dmg_transfer": dmg_tr,
            "flips_aegis": flips_aegis, "flips_blackbox": flips_bb, "flips_transfer": flips_tr,
            "transfer_kind": which_tr, "bb_kind": bb_kind,
            # per-attacker black-box damage at the same M-query budget
            "dmg_bb_rs": dmg_rs, "dmg_bb_simba": dmg_simba, "dmg_bb_nes": dmg_nes,
            "rs_frac": dmg_rs / max(dmg_aegis, 1e-12),
            "simba_frac": dmg_simba / max(dmg_aegis, 1e-12),
            "nes_frac": dmg_nes / max(dmg_aegis, 1e-12),
            # fraction of AEGIS damage reached by each independent attacker (<1 => AEGIS leads)
            "blackbox_frac": dmg_bb / max(dmg_aegis, 1e-12),
            "transfer_frac": dmg_tr / max(dmg_aegis, 1e-12),
            # AEGIS advantage (>1 => AEGIS leads)
            "aegis_over_blackbox": dmg_aegis / max(dmg_bb, 1e-12),
            "aegis_over_transfer": dmg_aegis / max(dmg_tr, 1e-12),
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [OOM] {ds_name} seed {seed} -- skipped", flush=True)
            return None
        raise


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  eps={EPS}  M={M}  surrogate=seed+{SURR_OFFSET}", flush=True)
    rows, t0 = [], time.time()
    for name, loader in load_datasets():
        data = loader()
        print(f"\n{name} (N={data['N']}):", flush=True)
        for seed in SEEDS:
            r = run_single(name, data, seed, device)
            if r:
                rows.append(r)
                print(f"  seed {seed}: AEGIS={r['dmg_aegis']:.3f}  "
                      f"BB(best/{M})={r['dmg_blackbox']:.3f}({r['blackbox_frac']:.2f})  "
                      f"Transfer={r['dmg_transfer']:.3f}({r['transfer_frac']:.2f},{r['transfer_kind']})  "
                      f"flips A/B/T={r['flips_aegis']}/{r['flips_blackbox']}/{r['flips_transfer']}",
                      flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        print("No rows produced (all cells skipped) -- nothing to save.", flush=True)
        return
    out = Path("results/x7_independent_attack.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nTotal: {time.time()-t0:.0f}s   Saved {out}\n" + "=" * 64, flush=True)

    # ---- summary: per-dataset fraction of AEGIS reached by best independent attack ----
    print(f"{'Dataset':10s} {'AEGIS dmg':>10} {'BB frac':>9} {'Transfer frac':>14} "
          f"{'best-indep frac':>16}", flush=True)
    print("-" * 64)
    all_bb, all_tr, all_best = [], [], []
    for name, _ in load_datasets():
        sub = [r for r in rows if r["dataset"] == name]
        if not sub:
            continue
        a = np.mean([r["dmg_aegis"] for r in sub])
        bb = np.mean([r["blackbox_frac"] for r in sub])
        tr = np.mean([r["transfer_frac"] for r in sub])
        best = np.mean([max(r["blackbox_frac"], r["transfer_frac"]) for r in sub])
        all_bb.append(bb); all_tr.append(tr); all_best.append(best)
        print(f"{name:10s} {a:10.3f} {bb:9.2f} {tr:14.2f} {best:16.2f}", flush=True)
    if all_best:
        print("-" * 64)
        print(f"{'MEAN':10s} {'':10} {np.mean(all_bb):9.2f} {np.mean(all_tr):14.2f} "
              f"{np.mean(all_best):16.2f}", flush=True)
        print(f"\nInterpretation: frac<1 => AEGIS's single query beats the independent attacker.")
        print(f"  black-box uses {M} queries; transfer uses an independently trained surrogate;")
        print(f"  AEGIS uses exactly 1 query (one S_c construction + SVD).")


if __name__ == "__main__":
    sys.exit(main() or 0)
