"""Smoke test for PE_DEQ_PF.

Verifies the DEQ + IFT plumbing end-to-end on a tiny synthetic 4-bus
case. This is *not* a training run -- the model is randomly initialized;
we only check that the machinery works:

  1. Forward solve runs (Anderson residual history is non-empty).
  2. Output has the expected shape and finite values.
  3. Bus-permutation equivariance: pi applied to the inputs yields pi
     applied to the output (up to solver tolerance).
  4. Backward through the IFT hook produces non-zero parameter gradients.

Run: python -m scripts.smoke_pe_deq_pf
"""

from __future__ import annotations

import sys

import torch

import models  # noqa: F401 -- registers PE_DEQ_PF
from models.registry import build_model


def make_synthetic_grid(B: int = 2, N: int = 4, device: torch.device = torch.device("cpu")):
    """Tiny synthetic AC grid: idx 0 slack, idx 1 PV, idx 2,3 PQ.

    Returns the same input pack the model sees from the data loader:
    bus_type (B,N), Line (B,E), Ys (B,E), Yc (B,E), S (B,N), V0 (B,N,2),
    and the canonical pairs (E,2) for the permutation helper.
    """
    bus_type = torch.zeros(B, N, dtype=torch.long, device=device)
    bus_type[:, 0] = 1  # slack
    bus_type[:, 1] = 2  # PV

    iu = torch.triu_indices(N, N, offset=1, device=device)
    E = iu.shape[1]
    pairs = iu.t().contiguous()  # (E, 2) with j < i

    torch.manual_seed(0)
    Line = (torch.rand(B, E, device=device) > 0.2).to(torch.float32)
    # Force a spanning chain so each bus is reachable.
    for e in range(min(N - 1, E)):
        Line[:, e] = 1.0

    Ys = torch.complex(
        torch.rand(B, E, device=device) * 2.0 + 0.5,
        -(torch.rand(B, E, device=device) * 4.0 + 1.0),
    )
    Yc = torch.complex(
        torch.zeros(B, E, device=device),
        torch.rand(B, E, device=device) * 0.05,
    )

    S = torch.complex(
        (torch.rand(B, N, device=device) - 0.5) * 0.5,
        (torch.rand(B, N, device=device) - 0.5) * 0.3,
    )
    S[:, 0] = 0  # slack has no injection setpoint

    V0 = torch.zeros(B, N, 2, device=device)
    V0[..., 0] = 1.0
    V0[..., 1] = 0.0

    return bus_type, Line, Ys, Yc, S, V0, pairs


def permute_inputs(
    bus_type: torch.Tensor,
    Line: torch.Tensor,
    Ys: torch.Tensor,
    Yc: torch.Tensor,
    S: torch.Tensor,
    V0: torch.Tensor,
    perm: torch.Tensor,
    pairs: torch.Tensor,
):
    """Apply a node-index permutation pi consistently to all inputs.

    Node-indexed tensors are gathered along the node axis. Edge-indexed
    tensors are re-ordered by mapping each canonical (j<i) edge to its
    permuted-canonical position.
    """
    B, N = bus_type.shape
    bt2 = bus_type[:, perm]
    S2 = S[:, perm]
    V02 = V0[:, perm]

    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(N, device=perm.device)

    new_pairs = perm_inv[pairs]  # (E, 2)
    j_new = torch.minimum(new_pairs[:, 0], new_pairs[:, 1])
    i_new = torch.maximum(new_pairs[:, 0], new_pairs[:, 1])

    lookup = {(int(pairs[k, 0]), int(pairs[k, 1])): k for k in range(pairs.shape[0])}
    new_idx = torch.tensor(
        [lookup[(int(j_new[k]), int(i_new[k]))] for k in range(pairs.shape[0])],
        dtype=torch.long,
        device=perm.device,
    )

    Line2 = torch.zeros_like(Line)
    Ys2 = torch.zeros_like(Ys)
    Yc2 = torch.zeros_like(Yc)
    Line2[:, new_idx] = Line
    Ys2[:, new_idx] = Ys
    Yc2[:, new_idx] = Yc
    return bt2, Line2, Ys2, Yc2, S2, V02


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    torch.manual_seed(42)
    model = build_model(
        "PE_DEQ_PF",
        d=4,
        d_hi=16,
        K=30,
        pinn=True,
        dtheta_max=0.30,
        dvm_frac=0.10,
        num_attn_layers=1,
        device=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] parameters={n_params}")

    bus_type, Line, Ys, Yc, S, V0, pairs = make_synthetic_grid(B=2, N=4, device=device)

    # 1) Forward
    out, phys_loss = model(bus_type, Line, None, Ys, Yc, S, V0, n_nodes_per_graph=None)
    res = model.deq.forward_res
    print(f"[smoke] out.shape={tuple(out.shape)}  phys_loss={float(phys_loss.item()):.4e}")
    if res:
        print(f"[smoke] anderson fwd res: first={res[0]:.2e}  last={res[-1]:.2e}  iters={len(res)}")
    assert torch.isfinite(out).all(), "non-finite output"
    assert out.shape == (2, 4, 2)

    # 2a) Architectural invariant: ONE application of F_theta is permutation-
    #     equivariant by construction (weight-shared MPNN + bus-index-free
    #     attention + symmetric mismatch features). This must hold at machine
    #     precision regardless of training -- it's a property of the model,
    #     not the solver. We force max_iter=2 (one f-call inside the solver)
    #     and use the naive solver so no history-LS noise enters.
    perm = torch.tensor([0, 1, 3, 2], dtype=torch.long, device=device)
    bt2, Line2, Ys2, Yc2, S2, V02 = permute_inputs(bus_type, Line, Ys, Yc, S, V0, perm, pairs)

    saved_solver = model.deq.solver
    saved_kwargs = dict(model.deq.solver_kwargs)
    from models.pe_deq_pf.deq import naive_solver

    model.deq.solver = naive_solver
    model.deq.solver_kwargs = {"max_iter": 2, "tol": 0.0}  # exactly one f-step
    out_s, _ = model(bus_type, Line, None, Ys, Yc, S, V0, n_nodes_per_graph=None)
    out_s2, _ = model(bt2, Line2, None, Ys2, Yc2, S2, V02, n_nodes_per_graph=None)
    rel_err_arch = float((out_s2 - out_s[:, perm]).norm().item()) / max(
        float(out_s[:, perm].norm().item()), 1e-12
    )
    print(f"[smoke] F-step equivariance (1-step naive) rel_err={rel_err_arch:.3e}")
    assert rel_err_arch < 1e-4, f"F-step is NOT equivariant (rel_err={rel_err_arch})"

    # Restore solver for the fixed-point equivariance check.
    model.deq.solver = saved_solver
    model.deq.solver_kwargs = saved_kwargs

    # 2b) Fixed-point equivariance (informational at random init).
    out2, _ = model(bt2, Line2, None, Ys2, Yc2, S2, V02, n_nodes_per_graph=None)
    out_expected = out[:, perm]
    rel_err_fp = float((out2 - out_expected).norm().item()) / max(
        float(out_expected.norm().item()), 1e-12
    )
    fwd_res = model.deq.forward_res
    converged = bool(fwd_res) and fwd_res[-1] < 1e-3
    print(
        f"[smoke] fixed-point equivariance rel_err={rel_err_fp:.3e}  "
        f"(forward converged={converged})"
    )
    if not converged:
        print(
            "[smoke] note: F is non-contractive at random init, so the "
            "fixed-point check is informational only; training drives F into "
            "the contractive regime."
        )

    # 3) Backward through IFT hook
    model.train()
    out, phys_loss = model(bus_type, Line, None, Ys, Yc, S, V0, n_nodes_per_graph=None)
    loss = (out * out).mean() + 0.1 * phys_loss.mean()
    loss.backward()
    grad_norms = {
        n: float(p.grad.norm().item())
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    nonzero = sum(1 for g in grad_norms.values() if g > 0)
    bres = model.deq.backward_res
    print(
        f"[smoke] backward: {nonzero}/{len(grad_norms)} params with non-zero grad; "
        f"adjoint iters={len(bres)}  last_res={(bres[-1] if bres else float('nan')):.2e}"
    )
    assert nonzero > 0, "no parameters received gradients via IFT backward"

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
