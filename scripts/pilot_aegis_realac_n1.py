"""PILOT: score AEGIS S_c (and graph-centrality baselines) against the REAL
PandaPower AC N-1 ground truth — not the surrogate's own edge-deletion.

Motivation: the headline PF table (`tab:ieee`) scores AEGIS against
`greedy_structural_attack` (the surrogate reconverged under edge deletion),
NOT against an independent AC contingency. This pilot computes the honest
number: AEGIS vs `true_n1_severity` (pp.runpp, line out-of-service, L2 of
ΔV,Δθ) — the same real-AC truth R2_05/R2_06 use for the PI/LODF baselines.

It ALSO scores degree / edge-betweenness / current-flow-betweenness
centrality vs the same real-AC truth, to test the topology-confound (R4).

CRITICAL: verifies the surrogate-bus-pair <-> pandapower-line-bus-pair mapping
(edge-set overlap) BEFORE trusting any tau. If the overlap is low the bus
orderings disagree and the numbers are meaningless.

Usage:
    .venv/bin/python scripts/pilot_aegis_realac_n1.py --cases case14,case57 --seeds 3
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401 -- register model builders
import networkx as nx
import pandapower as pp
import pandapower.networks as pp_nets

from iem.adversarial import (
    _compute_structural_jacobian,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.contractive_pf import ContractiveGCN_PF
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

CASE_FN = {
    "case14": pp_nets.case14,
    "case30": pp_nets.case30,
    "case57": pp_nets.case57,
    "case118": pp_nets.case118,
}
DS_PATH = {c: f"datasets/IEEE_{c}_2000.parquet" for c in CASE_FN}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def key(i, j):
    return (min(int(i), int(j)), max(int(i), int(j)))


# --------------------------------------------------------------------------- #
# Surrogate side: train + AEGIS S_c per-edge ranking (mirrors run_single)
# --------------------------------------------------------------------------- #
def compute_aegis(case_name, ds_path, seed, device):
    set_seed(seed)
    if not Path(ds_path).exists():
        return None
    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ep in range(30):
        model.train()
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()

    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)))
    with torch.no_grad():
        V_pred, ctx_pf = model(
            batch["bus_type"].to(device), batch["Lines_connected"].to(device),
            None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device), batch["V_start"].to(device),
            batch["sizes"].to(device),
        )
    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    X_proj = ctx_pf["X_proj"]
    N = int(batch["sizes"][0].item())
    A_sub = A_hat[:N, :N]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj[:N]}

    Z = Z_star[:N].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    J_z, J_A, _ = _compute_structural_jacobian(lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    aegis = {key(i, j): float(v) for i, j, v in attack["all_edge_vulnerabilities"]}
    return aegis, set(aegis.keys()), N


# --------------------------------------------------------------------------- #
# Pandapower side: REAL AC N-1 severity + topology centralities
# --------------------------------------------------------------------------- #
def real_ac_n1(net):
    """{(min,max) bus pair: L2(ΔV,Δθ)} per single-line outage via AC power flow."""
    base = copy.deepcopy(net)
    pp.runpp(base)
    Vb = base.res_bus["vm_pu"].values.copy()
    Tb = np.deg2rad(base.res_bus["va_degree"].values.copy())
    sev = {}
    for li in net.line.index:
        n2 = copy.deepcopy(net)
        n2.line.at[li, "in_service"] = False
        try:
            pp.runpp(n2)
            Vp = n2.res_bus["vm_pu"].values
            Tp = np.deg2rad(n2.res_bus["va_degree"].values)
            s = float(np.sqrt(np.sum((Vp - Vb) ** 2) + np.sum((Tp - Tb) ** 2)))
            if not np.isfinite(s):
                s = float("inf")  # converged with islanded NaN buses = most severe
        except Exception:
            s = float("inf")  # islanding / divergence = most severe
        fb, tb = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
        sev[key(fb, tb)] = s
    # rank inf (diverged) as most-severe via a large finite sentinel
    finite = [v for v in sev.values() if np.isfinite(v)]
    big = (max(finite) * 10.0) if finite else 1.0
    return {k: (v if np.isfinite(v) else big) for k, v in sev.items()}


def build_graph(net):
    G = nx.Graph()
    G.add_nodes_from([int(b) for b in net.bus.index])
    for li in net.line.index:
        G.add_edge(int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"]))
    for ti in net.trafo.index:
        G.add_edge(int(net.trafo.at[ti, "hv_bus"]), int(net.trafo.at[ti, "lv_bus"]))
    return G


def centrality_rankings(net):
    G = build_graph(net)
    deg = dict(G.degree())
    degree_edge = {key(u, v): float(deg[u] + deg[v]) for u, v in G.edges()}
    ebet = {key(u, v): float(s) for (u, v), s in nx.edge_betweenness_centrality(G).items()}
    try:
        cfb = {key(u, v): float(s) for (u, v), s in nx.edge_current_flow_betweenness_centrality(G).items()}
    except Exception:
        cfb = None
    return {"degree": degree_edge, "edge_betweenness": ebet, "current_flow_betweenness": cfb}


# --------------------------------------------------------------------------- #
# Scoring vs real-AC ground truth (on the common line-edge set)
# --------------------------------------------------------------------------- #
def score(method_dict, realac, kmax=10):
    if method_dict is None:
        return None, None, 0
    keys = [k for k in realac.keys() if k in method_dict]
    if len(keys) < 3:
        return None, None, len(keys)
    m = np.array([method_dict[k] for k in keys])
    t = np.array([realac[k] for k in keys])
    if m.std() == 0 or t.std() == 0:
        tau = float("nan")
    else:
        tau, _ = kendalltau(m, t)
    k = min(kmax, len(keys))
    top_m = set(np.argsort(-m)[:k].tolist())
    top_t = set(np.argsort(-t)[:k].tolist())
    return float(tau), len(top_m & top_t) / k, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="case14,case57")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cases = args.cases.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cases={cases}  seeds={args.seeds}\n", flush=True)

    methods = ["aegis", "degree", "edge_betweenness", "current_flow_betweenness"]
    agg = {c: {m: {"tau": [], "p10": []} for m in methods} for c in cases}

    for case_name in cases:
        net = CASE_FN[case_name]()
        realac = real_ac_n1(net)
        cents = centrality_rankings(net)
        n_lines = len(realac)

        for s_idx in range(args.seeds):
            seed = SEEDS[s_idx]
            t0 = time.time()
            res = compute_aegis(case_name, DS_PATH[case_name], seed, device)
            if res is None:
                print(f"  {case_name}: dataset missing -> SKIP", flush=True)
                break
            aegis, surro_edges, N = res

            # --- MAPPING VERIFICATION (count-test) ---
            common = set(realac.keys()) & surro_edges
            overlap = len(common) / max(1, n_lines)
            if s_idx == 0:
                print(f"[{case_name}] N={N}  pp_lines={n_lines}  surrogate_edges={len(surro_edges)}  "
                      f"line<->surrogate overlap={overlap:.2f}", flush=True)
                if overlap < 0.9:
                    print(f"  !! WARNING overlap<0.9 — bus orderings likely DISAGREE; tau is untrustworthy", flush=True)

            method_dicts = {"aegis": aegis, **cents}
            line = []
            for m in methods:
                tau, p10, ncmp = score(method_dicts[m], realac)
                if tau is not None:
                    agg[case_name][m]["tau"].append(tau)
                    agg[case_name][m]["p10"].append(p10)
                line.append(f"{m}:tau={tau if tau is None else round(tau,3)},P@10={p10 if p10 is None else round(p10,2)}")
            print(f"  {case_name} seed={seed} ({time.time()-t0:.0f}s) [{ncmp} edges] " + " | ".join(line), flush=True)

    # --- summary ---
    print("\n" + "=" * 78)
    print("AEGIS & centrality vs REAL PandaPower AC N-1 (Kendall tau, P@10)")
    print("=" * 78)
    print(f"{'case':<9}{'method':<26}{'tau':>16}{'P@10':>12}")
    print("-" * 78)
    for c in cases:
        for m in methods:
            taus = agg[c][m]["tau"]
            p10s = agg[c][m]["p10"]
            if not taus:
                continue
            ts = f"{np.mean(taus):+.3f}±{np.std(taus):.3f}"
            ps = f"{np.mean(p10s):.2f}±{np.std(p10s):.2f}"
            print(f"{c:<9}{m:<26}{ts:>16}{ps:>12}")
    print("\nReference: headline AEGIS-vs-SURROGATE tau = case14 +0.42 / case57 +0.67")
    print("Reference: LODF/PI vs real-AC tau ~ +0.30-0.33 (R2_06/R2_05)")


if __name__ == "__main__":
    sys.exit(main() or 0)
