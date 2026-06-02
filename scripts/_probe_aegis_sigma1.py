#!/usr/bin/env python3
"""BULLETPROOF probe: validate aegis_sigma1 (matrix-free, Neumann + power
iteration + autograd adjoint) against a DENSE ground-truth S_c on a tiny IGNN.

If the matrix-free sigma_1 matches torch.linalg.svdvals of the explicitly-formed
S_c = (I - J_z)^{-1} J_A P_c, then the operator, J_z/J_A Jacobian actions, the
Neumann inverse, the power iteration AND the Sc_rmatvec adjoint are all correct
end-to-end (a wrong adjoint would make power iteration converge to the wrong
singular vector and miss sigma_1). float32 throughout to match the real run
path exactly. CPU-only, tiny -> no GPU contention.
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/redwanul/Storage/Work/PR-LAB/GNN_load_flow/GNN_load_flow/GNN/SimpleGNN")
sys.path.insert(0, str(ROOT))

from iem.examples.ignn_cora import IGNN
from scripts.exp_aegis_regularized_training import (
    aegis_sigma1, _active_edge_index,
)

torch.manual_seed(0)
N, nf, hid, nc = 10, 5, 4, 3

# sparse symmetric A_hat (exact zeros so the active-edge set is well defined)
A = torch.zeros(N, N)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5),
         (5, 6), (6, 7), (7, 8), (8, 9), (9, 1), (2, 7)]
for i, j in edges:
    w = 0.3 + 0.1 * ((i + j) % 3)
    A[i, j] = w
    A[j, i] = w
X = torch.randn(N, nf)

model = IGNN(nf, hidden=hid, n_classes=nc, c=0.9, dropout=0.0)
model.eval()
ctx = {"A_hat": A, "X_proj": model.U(X)}

# --- equilibrium z* (Picard) ---
Z = torch.zeros(N, hid)
for _ in range(2000):
    Zn = model.operator(Z, ctx)
    if (Zn - Z).norm() < 1e-7:
        Z = Zn
        break
    Z = Zn
zstar = Z.detach()


def F_of_z(zf):
    return model.operator(zf.reshape(N, hid), ctx).reshape(-1)


def F_of_A(Av):
    return model.operator(zstar, {"A_hat": Av, "X_proj": model.U(X)}).reshape(-1)


Nh = N * hid
Jz = torch.autograd.functional.jacobian(F_of_z, zstar.reshape(-1))          # (Nh, Nh)
JA = torch.autograd.functional.jacobian(F_of_A, A)                          # (Nh, N, N)

edge_idx = _active_edge_index(A)
E = edge_idx.shape[0]
JAPc = torch.zeros(Nh, E)
for k in range(E):
    i, j = edge_idx[k].tolist()
    JAPc[:, k] = JA[:, i, j] + JA[:, j, i]                                  # P_c: both (i,j),(j,i)

Ieye = torch.eye(Nh)
Sc = torch.linalg.solve(Ieye - Jz, JAPc)                                    # (Nh, E) EXACT inverse
svals = torch.linalg.svdvals(Sc)
sig_true = float(svals[0])
rho_Jz = float(torch.linalg.eigvals(Jz).abs().max())
opn_Jz = float(torch.linalg.svdvals(Jz)[0])

# matrix-free estimate (Neumann K, power-iter), frozen z* to match the GT linearization
sig_est = float(aegis_sigma1(model, X, A, K_neumann=60, n_power=40, detach_zstar=True))

rel = abs(sig_true - sig_est) / (abs(sig_true) + 1e-30) * 100.0
gap = float(svals[0] / (svals[1] + 1e-30)) if E > 1 else float("inf")

print(f"PROBE: N={N} hid={hid} edges={E} rho(Jz)={rho_Jz:.4f} ||Jz||2={opn_Jz:.4f}")
print(f"  sigma1 DENSE-SVD (ground truth) = {sig_true:.6f}")
print(f"  sigma1 aegis (matrix-free)      = {sig_est:.6f}")
print(f"  relative error                  = {rel:.4f}%   (sigma1/sigma2 gap={gap:.2f})")
print(f"  VERDICT: {'PASS (machinery correct)' if rel < 1.0 else 'FAIL (bug in matrix-free path)'}")
