"""Permutation-Equivariant Deep-Equilibrium AC-PF model (PE_DEQ_PF).

Replaces the K-step line-search post-correction in PIGNN-Attn-LS with a
single weight-tied operator F_theta whose fixed point is the AC-PF
solution. Forward solve via Anderson acceleration; backward via the
implicit function theorem.

State per node: z = (v, theta, m) packed along the last dim, total size
2 + d. The forward signature matches GNSMsg_EdgeSelfAttn so the existing
training pipeline is unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse grid-graph utilities and the attention block from edge_selfattn.
from models.edge_selfattn.admittance import (
    build_dense_Y,
    build_edges_blockdiag,
    build_edges_plain,
)
from models.edge_selfattn.attention import EdgeSelfAttnBlock

from .deq import DEQFixedPoint, anderson_solver, jacobian_reg_estimate, naive_solver


class PE_DEQ_PF(nn.Module):
    """Weight-tied DEQ surrogate for AC-PF.

    Architecture: a single edge-attention block, fed mismatch features,
    produces residual updates (dv, dth, dm) damped by a learnable scalar
    alpha in (0, 1). The fixed point z_star = F_theta(z_star) corresponds
    to zero residual updates -- i.e. KCL-balanced (V, theta).

    Permutation equivariance: weight-shared MPNN + bus-index-free message
    function => exact equivariance under bus relabeling.
    """

    def __init__(
        self,
        d: int = 4,
        d_hi: int = 16,
        d_model: int | None = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        v_min: float = 0.75,
        v_max: float = 1.20,
        pinn: bool = True,
        solver: str = "anderson",
        forward_iter: int = 30,
        backward_iter: int = 30,
        forward_tol: float = 1e-4,
        backward_tol: float = 1e-6,
        anderson_m: int = 5,
        anderson_lam: float = 1e-4,
        anderson_beta: float = 1.0,
        damping_init: float = 0.1,
        backward_mode: str = "phantom",
        jac_reg_weight: float = 0.0,
        jac_reg_n_samples: int = 1,
        unrolled_warmup_epochs: int = 0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.d, self.d_hi = int(d), int(d_hi)
        self.d_model = int(d_model if d_model is not None else d_hi)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_attn_layers = int(num_attn_layers)
        self.pinn = bool(pinn)

        self.dtheta_max = float(dtheta_max)
        self.dvm_frac = float(dvm_frac)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.bus_feat_dim = 4 + self.d  # [v, theta, dP, dQ] + m
        self.edge_feat_dim = 4  # [Ysr, Ysi, Yc_real, Yc_imag]

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [
                EdgeSelfAttnBlock(
                    self.d_model,
                    self.n_heads,
                    self.edge_feat_dim,
                    ffn_hidden=4 * self.d_model,
                    dropout=attn_dropout,
                )
                for _ in range(self.num_attn_layers)
            ]
        )

        # Single weight-tied head set (vs. K sets in PIGNN-Attn-LS).
        self.theta_head = nn.Linear(self.d_model, 1)
        self.v_head = nn.Linear(self.d_model, 1)
        self.m_head = nn.Linear(self.d_model, self.d)

        # Small random head init (std=0.02) -- NOT zero -- so that F_theta
        # is not trivially the identity at random init. The adjoint
        # (I - dF/dz)^T system is singular if dF/dz = I, which is what
        # zero-init heads produce; perturbing the heads makes the IFT
        # backward well-conditioned from step 1. Biases stay at zero.
        for h in (self.theta_head, self.v_head, self.m_head):
            nn.init.normal_(h.weight, mean=0.0, std=0.02)
            nn.init.zeros_(h.bias)

        # Learnable damping in (0, 1) replaces Armijo line-search step size.
        damping_init = float(damping_init)
        if not 0.0 < damping_init < 1.0:
            raise ValueError("damping_init must be in (0, 1)")
        logit = math.log(damping_init / (1.0 - damping_init))
        self.damping_logit = nn.Parameter(torch.tensor(float(logit)))

        solver_map = {"anderson": anderson_solver, "naive": naive_solver}
        if solver not in solver_map:
            raise ValueError(f"Unknown solver: {solver!r}")
        self._solver_name = solver
        if solver == "anderson":
            solver_kwargs = {
                "m": int(anderson_m),
                "lam": float(anderson_lam),
                "beta": float(anderson_beta),
                "max_iter": int(forward_iter),
                "tol": float(forward_tol),
            }
        else:
            solver_kwargs = {"max_iter": int(forward_iter), "tol": float(forward_tol)}
        backward_kwargs = dict(solver_kwargs)
        backward_kwargs["max_iter"] = int(backward_iter)
        backward_kwargs["tol"] = float(backward_tol)

        self.deq = DEQFixedPoint(
            f=self._operator,
            solver=solver_map[solver],
            solver_kwargs=solver_kwargs,
            backward_solver=solver_map[solver],
            backward_kwargs=backward_kwargs,
            backward_mode=backward_mode,
        )

        # Jacobian regularization (Bai+Koltun+Kolter 2021). When > 0, an
        # auxiliary penalty E[||dF/dz||_F^2] is added to phys_loss during
        # training. Drives the operator toward contractivity, which is
        # what makes the IFT backward well-conditioned.
        self.jac_reg_weight = float(jac_reg_weight)
        self.jac_reg_n_samples = int(jac_reg_n_samples)
        if self.jac_reg_weight < 0:
            raise ValueError("jac_reg_weight must be >= 0")

        self._pair_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        # Curriculum: during the first `unrolled_warmup_epochs`, use a fully
        # explicit K-step unrolled forward (BPTT over K iterations) with no
        # DEQ solver. After warmup, switch to the implicit DEQ forward+backward.
        # Bypasses the chaotic pre-contractive phase that plagued the DEQ-only
        # runs (C, C100, D).
        self.unrolled_warmup_epochs = int(unrolled_warmup_epochs)
        self._current_epoch: int = 0
        # forward_iter is the K used for both the DEQ solver max_iter and the
        # unrolled warmup steps -- they share the same notion of "iterations".
        self._unrolled_K: int = int(forward_iter)

        # Spectral normalization: wraps the attention block's Linear weights
        # so that ||W||_2 <= 1, which gives a Lipschitz-by-construction prior
        # on F's local sensitivity. Combined with bounded damping alpha, this
        # is the standard contractivity-by-construction recipe.
        if spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for module in self.blocks.modules():
                if isinstance(module, nn.Linear):
                    _sn(module, name="weight", n_power_iterations=1)
        self._spectral_norm_on = bool(spectral_norm)

    def set_epoch(self, epoch: int) -> None:
        """Called by the training loop before each train epoch.

        Used by the curriculum: forward() checks `self._current_epoch` to
        decide between unrolled-warmup and DEQ paths.
        """
        self._current_epoch = int(epoch)

    @property
    def in_warmup(self) -> bool:
        return self._current_epoch < self.unrolled_warmup_epochs

    @property
    def damping(self) -> torch.Tensor:
        return torch.sigmoid(self.damping_logit)

    @torch.no_grad()
    def _pairs_for_n(self, n: int, device: torch.device) -> torch.Tensor:
        key = (int(n), device)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        iu = torch.triu_indices(int(n), int(n), offset=1, device=device)
        pairs = iu.t().contiguous()
        self._pair_cache[key] = pairs
        return pairs

    def _wrap_theta(self, th: torch.Tensor) -> torch.Tensor:
        return (th + math.pi) % (2 * math.pi) - math.pi

    def _apply_blocks(
        self,
        x: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat_dir: torch.Tensor,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, edge_index_dir, edge_feat_dir)
        return x

    def _operator(self, z: torch.Tensor, ctx: dict) -> torch.Tensor:
        """Single application of F_theta. Returns next state z'.

        Theta is NOT wrapped inside the iteration (state stays smooth for
        Anderson). Wrap is applied only at readout time in forward().
        """
        B, N = ctx["B"], ctx["N"]
        v = z[..., 0]
        th = z[..., 1]
        m = z[..., 2:]  # (B, N, d)

        Y = ctx["Y"]
        P_set, Q_set = ctx["P_set"], ctx["Q_set"]
        slack_mask = ctx["slack_mask"]
        pv_mask = ctx["pv_mask"]

        # KCL mismatch features at current state
        Vc = v * torch.exp(1j * th)
        Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
        Sc = Vc * Ic.conj()
        DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
        DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

        bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
        x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

        if ctx["block_diag"]:
            x = self._apply_blocks(x, ctx["edge_index_dir"], ctx["edge_feat_dir"])
        else:
            x_out = x.clone()
            for b in range(B):
                e_b = ctx["edge_index_dir_list"][b]
                if e_b.numel() == 0:
                    continue
                xb = self._apply_blocks(x[b : b + 1], e_b, ctx["edge_feat_dir_list"][b])
                x_out[b : b + 1] = xb
            x = x_out

        dth = self.theta_head(x).squeeze(-1)
        dv = self.v_head(x).squeeze(-1)
        dm = torch.tanh(self.m_head(x))
        dm = F.layer_norm(dm, dm.shape[-1:])

        dth = dth.masked_fill(slack_mask, 0.0)
        dv = dv.masked_fill(slack_mask | pv_mask, 0.0)
        v_abs = v.abs().detach()
        dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
        dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

        alpha = self.damping
        v_new = torch.clamp(v + alpha * dv, self.v_min, self.v_max)
        th_new = th + alpha * dth
        m_new = m + alpha * dm

        return torch.cat(
            [v_new.unsqueeze(-1), th_new.unsqueeze(-1), m_new], dim=-1
        )

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph):
        device = bus_type.device
        B, N = bus_type.shape

        ctx: dict = {"B": B, "N": N}

        if n_nodes_per_graph is not None:
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d = Ys.squeeze(0)
            Yc_1d = Yc.squeeze(0)
            (
                undirected,
                _,
                edge_index_dir,
                edge_feat_dir,
                ys_edge,
                yc_edge,
            ) = build_edges_blockdiag(
                line_mask_1d=Line_1d,
                Ys_1d=Ys_1d,
                Yc_1d=Yc_1d,
                n_nodes_per_graph=n_nodes_per_graph,
                edge_feat_dim=self.edge_feat_dim,
                pairs_for_n=self._pairs_for_n,
                device=device,
            )
            if Y is None:
                Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)
            ctx["block_diag"] = True
            ctx["edge_index_dir"] = edge_index_dir
            ctx["edge_feat_dir"] = edge_feat_dir
        else:
            pairs = self._pairs_for_n(N, device)
            (
                edge_index_dir_list,
                edge_feat_dir_list,
                undirected_list,
                mask_list,
            ) = build_edges_plain(
                Line=Line,
                Ys=Ys,
                Yc=Yc,
                N=N,
                edge_feat_dim=self.edge_feat_dim,
                pairs=pairs,
                device=device,
            )
            if Y is None:
                Y_list = []
                for b in range(B):
                    mask = mask_list[b]
                    undirected = undirected_list[b]
                    ys_edge_b = Ys[b][mask]
                    yc_edge_b = Yc[b][mask]
                    Y_list.append(
                        build_dense_Y(N, undirected, ys_edge_b, yc_edge_b, device=device)
                    )
                Y = torch.stack(Y_list, dim=0)
            ctx["block_diag"] = False
            ctx["edge_index_dir_list"] = edge_index_dir_list
            ctx["edge_feat_dir_list"] = edge_feat_dir_list

        slack_mask = bus_type == 1
        pv_mask = bus_type == 2

        ctx["Y"] = Y
        ctx["P_set"] = S.real
        ctx["Q_set"] = S.imag
        ctx["slack_mask"] = slack_mask
        ctx["pv_mask"] = pv_mask

        v0 = V0[..., 0]
        th0 = V0[..., 1]
        m0 = v0.new_zeros(B, N, self.d)
        z0 = torch.cat(
            [v0.unsqueeze(-1), th0.unsqueeze(-1), m0], dim=-1
        )

        # Curriculum branch: explicit K-step unrolled forward during warmup
        # epochs (BPTT over K iterations), implicit DEQ forward+backward after.
        # The branch is on epoch, not on self.training, so validation and
        # training use the same forward at each stage.
        if self.in_warmup:
            z = z0
            for _ in range(self._unrolled_K):
                z = self._operator(z, ctx)
            z_star = z
        else:
            z_star = self.deq(z0, ctx)

        v_star = z_star[..., 0]
        th_star = self._wrap_theta(z_star[..., 1])
        out = torch.stack([v_star, th_star], dim=-1)

        if self.pinn:
            Vc = v_star * torch.exp(1j * th_star)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()
            DP = (ctx["P_set"] - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (ctx["Q_set"] - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)
            phys_loss = (DP * DP + DQ * DQ).mean().unsqueeze(0)

            # Optional Jacobian regularization, computed only in training.
            # Evaluated at z_star (fresh autograd graph) so the penalty is
            # on F's local Lipschitz behaviour at the fixed point.
            if self.training and self.jac_reg_weight > 0.0:
                z_in = z_star.detach().requires_grad_(True)
                f_out = self._operator(z_in, ctx)
                jac = jacobian_reg_estimate(f_out, z_in, n_samples=self.jac_reg_n_samples)
                phys_loss = phys_loss + self.jac_reg_weight * jac.unsqueeze(0)
            return out, phys_loss
        return out
