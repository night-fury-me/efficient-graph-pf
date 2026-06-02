"""Part (d) downstream smoke test: confirm exp_fullgraph_attack_table imports and
a single (Cora, 1 seed) row runs without error on the new c=0.9 model.

We import the module (which transitively imports exp_full_attack_table.train_ignn
-> IGNN with the new c=0.9/dropout defaults) and call run_single once on Cora
with do_shift=False (cheap: SVD + Cls-PGD + Random attacks only).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Import path matches how the script is normally run (python -m / cwd=root).
import scripts.exp_fullgraph_attack_table as fg  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"imported exp_fullgraph_attack_table OK | device={device} EPS={fg.EPS}",
          flush=True)
    datasets = fg.load_datasets()
    data = datasets["Cora"]
    seed = fg.SEEDS[0]
    # Confirm the model the downstream train_ignn builds is the new c=0.9 one.
    m = fg.train_ignn(data, device, seed)
    print(f"downstream IGNN: c={getattr(m, 'c', 'NA')} dropout={getattr(m, 'dropout', 'NA')}",
          flush=True)
    t0 = time.time()
    r = fg.run_single("Cora", data, seed, fg.EPS, device, do_shift=False)
    dt = time.time() - t0
    if r is None:
        print("run_single returned None (no edges) -- UNEXPECTED", flush=True)
        print("DOWNSTREAM_FAIL", flush=True)
        return
    print(f"run_single(Cora, seed={seed}) OK [{dt:.0f}s]", flush=True)
    print(f"  rho={r['rho']:.3f} neumann_K={r['neumann_K']} sigma1={r['sigma1']:.2f}",
          flush=True)
    print(f"  SVD dmg={r['dmg_svd']:.3f} flips={r['flips_svd']} | "
          f"ClsPGD dmg={r['dmg_cls_pgd']:.3f} flips={r['flips_cls_pgd']} | "
          f"Random dmg={r['dmg_random']:.3f} flips={r['flips_random']}", flush=True)
    finite = all(
        (r[k] == r[k]) and r[k] not in (float("inf"), float("-inf"))
        for k in ["rho", "sigma1", "dmg_svd", "dmg_cls_pgd", "dmg_random"]
    )
    print(f"  ALL FINITE = {finite}", flush=True)
    print("DOWNSTREAM_OK" if finite else "DOWNSTREAM_FAIL", flush=True)


if __name__ == "__main__":
    main()
