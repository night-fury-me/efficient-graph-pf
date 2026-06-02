"""PILOT 2: can AEGIS recover REAL AC N-1 if we add a principled islanding term?

Failure mode on meshed grids (case57): the most-severe N-1 events are radial-
line DISCONNECTIONS (islanding), invisible to any continuous first-order
sensitivity (LODF included). Fix: augment the S_c flow score with a topological
islanding term — bridge edges (removal disconnects) ranked by island size.

Tests two questions:
  (1) FULL: does AEGIS+islanding recover real AC N-1 (vs degree/betweenness+islanding)?
  (2) NON-BRIDGE only: does S_c beat degree/betweenness on the FLOW edges
      (i.e. does S_c add real-physics value beyond topology)?

Usage:
    .venv/bin/python scripts/pilot_hybrid_islanding.py --cases case14,case57 --seeds 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_aegis_realac_n1 as P  # reuse verified harness
import torch


def bridge_island(G):
    """{(min,max) edge: island_size (smaller component) if bridge else 0}."""
    bridges = {P.key(u, v) for u, v in nx.bridges(G)}
    out = {}
    for u, v in list(G.edges()):
        k = P.key(u, v)
        if k in bridges:
            G.remove_edge(u, v)
            sizes = sorted(len(c) for c in nx.connected_components(G))
            G.add_edge(u, v)
            out[k] = sizes[0] if len(sizes) > 1 else 0
        else:
            out[k] = 0
    return out


def centrality_flow(G):
    """Topology flow-scores keyed by (min,max) edge: degree-sum, edge-betweenness."""
    deg = dict(G.degree())
    degree_edge = {P.key(u, v): float(deg[u] + deg[v]) for u, v in G.edges()}
    ebet = {P.key(u, v): float(s) for (u, v), s in nx.edge_betweenness_centrality(G).items()}
    return {"degree": degree_edge, "betweenness": ebet}


def hybrid(flow_dict, island_dict):
    """Bridges (by island size) ranked above all non-bridges (by normalized flow)."""
    keys = list(flow_dict.keys())
    fv = np.array([flow_dict[k] for k in keys], float)
    fn = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
    out = {}
    for idx, k in enumerate(keys):
        isl = island_dict.get(k, 0)
        out[k] = (10.0 + float(isl)) if isl > 0 else float(fn[idx])
    return out


def score_subset(method_dict, realac, allowed=None, kmax=10):
    realac2 = realac if allowed is None else {k: v for k, v in realac.items() if k in allowed}
    return P.score(method_dict, realac2, kmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="case14,case57")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    cases = args.cases.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cases={cases}  seeds={args.seeds}\n", flush=True)

    for case_name in cases:
        net = P.CASE_FN[case_name]()
        realac = P.real_ac_n1(net)
        G = P.build_graph(net)
        island = bridge_island(G)
        cents = centrality_flow(G)
        n_bridges_lines = sum(1 for k in realac if island.get(k, 0) > 0)
        nonbridge = {k for k in realac if island.get(k, 0) == 0}

        # AEGIS flow score (avg over seeds for stability of the topology comparison)
        aegis_acc = {}
        for s_idx in range(args.seeds):
            res = P.compute_aegis(case_name, P.DS_PATH[case_name], P.SEEDS[s_idx], device)
            if res is None:
                print(f"{case_name}: dataset missing -> SKIP")
                break
            aegis, _, _ = res
            for k, v in aegis.items():
                aegis_acc.setdefault(k, []).append(v)
        if not aegis_acc:
            continue
        aegis = {k: float(np.mean(v)) for k, v in aegis_acc.items()}

        flows = {"aegis": aegis, **cents}

        print(f"=== {case_name} ===  lines={len(realac)}  islanding(bridge) lines={n_bridges_lines}  "
              f"non-bridge lines={len(nonbridge)}", flush=True)
        print(f"{'method':<14}{'FULL tau':>11}{'FULL P@10':>11}   "
              f"{'+island tau':>12}{'+island P@10':>13}   {'nonbridge tau':>14}", flush=True)
        # bridge-only baseline (flow constant -> only island term ranks)
        bonly = hybrid({k: 0.0 for k in realac}, island)
        bt, bp, _ = score_subset(bonly, realac)
        print(f"{'island-only':<14}{'-':>11}{'-':>11}   {bt:>12.3f}{bp:>13.2f}   {'-':>14}", flush=True)
        for m, fd in flows.items():
            ft, fp, _ = score_subset(fd, realac)                       # flow only, full set
            ht, hp, _ = score_subset(hybrid(fd, island), realac)       # flow + islanding
            nt, _, nn = score_subset(fd, realac, allowed=nonbridge)    # flow only, non-bridge edges
            ft = -9 if ft is None else ft
            nt = -9 if nt is None else nt
            print(f"{m:<14}{ft:>11.3f}{fp:>11.2f}   {ht:>12.3f}{hp:>13.2f}   {nt:>14.3f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
