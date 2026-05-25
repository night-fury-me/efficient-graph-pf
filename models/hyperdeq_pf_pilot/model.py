"""HyperDEQ_PF_Pilot v2 -- PE_DEQ_PF with multi-point FiLM conditioning.

Architectural diff from PE_DEQ_PF:
1. FiLMHypernet (v2 - multi-point) consumes a 12-dim per-graph descriptor.
2. forward() computes the descriptor and per-point FiLM (gamma, beta) up front.
3. _operator applies FiLM after in_proj, and an overridden _apply_blocks_with_film
   applies FiLM after each attention block.

v1 was FiLM only after in_proj. v2 adds FiLM after each attention block so the
attention representations stay voltage-class-aware all the way to the heads.

For block_diag batches, FiLM (G, d_model) is broadcast over the buses
belonging to each sub-graph via repeat_interleave.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.edge_selfattn.admittance import (
    build_dense_Y,
    build_edges_blockdiag,
    build_edges_plain,
)
from models.pe_deq_pf.model import PE_DEQ_PF

from .descriptor import per_graph_descriptor
from .hypernet import FiLMHypernet


class HyperDEQ_PF_Pilot(PE_DEQ_PF):
    """Pilot variant: PE_DEQ_PF + per-block FiLM-conditioning."""

    def __init__(self, *args, hypernet_hidden: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        # n_points = 1 (in_proj) + num_attn_layers (after each attn block)
        self.n_film_points = 1 + int(self.num_attn_layers)
        self.hypernet = FiLMHypernet(
            d_model=self.d_model,
            n_points=self.n_film_points,
            hidden=int(hypernet_hidden),
        )

    # --- helper: broadcast (G, d_model) gamma/beta to per-bus shape ---------
    @staticmethod
    def _broadcast_film(gamma_or_beta: torch.Tensor, ctx: dict) -> torch.Tensor:
        """For block_diag, returns (1, M, d_model). For plain, (B, 1, d_model)."""
        if ctx["block_diag"]:
            sizes = ctx["sizes"]
            return torch.repeat_interleave(gamma_or_beta, sizes, dim=0).unsqueeze(0)
        else:
            return gamma_or_beta.unsqueeze(1)

    # --- attention with FiLM-after-each-block --------------------------------
    def _apply_blocks_with_film(
        self,
        x: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_feat_dir: torch.Tensor,
        film_blocks: list[tuple[torch.Tensor, torch.Tensor]],
        ctx: dict,
    ) -> torch.Tensor:
        for blk, (gamma, beta) in zip(self.blocks, film_blocks):
            x = blk(x, edge_index_dir, edge_feat_dir)
            g_bcast = self._broadcast_film(gamma, ctx)
            b_bcast = self._broadcast_film(beta, ctx)
            x = g_bcast * x + b_bcast
        return x

    # --- per-iteration operator with FiLM injection --------------------------
    def _operator(self, z: torch.Tensor, ctx: dict) -> torch.Tensor:
        B, N = ctx["B"], ctx["N"]
        v = z[..., 0]
        th = z[..., 1]
        m = z[..., 2:]

        Y = ctx["Y"]
        P_set, Q_set = ctx["P_set"], ctx["Q_set"]
        slack_mask = ctx["slack_mask"]
        pv_mask = ctx["pv_mask"]

        Vc = v * torch.exp(1j * th)
        Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
        Sc = Vc * Ic.conj()
        DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
        DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

        bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
        x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

        # --- FiLM point 0: after in_proj -----------------------------------
        gamma_in, beta_in = ctx["film_in"]
        g_b = self._broadcast_film(gamma_in, ctx)
        b_b = self._broadcast_film(beta_in, ctx)
        x = g_b * x + b_b

        if ctx["block_diag"]:
            x = self._apply_blocks_with_film(
                x, ctx["edge_index_dir"], ctx["edge_feat_dir"],
                ctx["film_blocks"], ctx,
            )
        else:
            x_out = x.clone()
            for b in range(B):
                e_b = ctx["edge_index_dir_list"][b]
                if e_b.numel() == 0:
                    continue
                # Slice per-graph FiLM for the plain (non-block_diag) path.
                film_blocks_b = [
                    (gamma[b : b + 1], beta[b : b + 1])
                    for (gamma, beta) in ctx["film_blocks"]
                ]
                xb = self._apply_blocks_with_film(
                    x[b : b + 1], e_b, ctx["edge_feat_dir_list"][b],
                    film_blocks_b, {**ctx, "block_diag": False},
                )
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

    # --- forward with descriptor + per-point FiLM precomputation -------------
    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph, **_unused):
        device = bus_type.device
        B, N = bus_type.shape

        g = per_graph_descriptor(
            bus_type=bus_type,
            Line=Line,
            Ys=Ys,
            S_start=S,
            V_start=V0,
            n_nodes_per_graph=n_nodes_per_graph,
        ).to(device)  # (G, 12)

        film_list = self.hypernet(g)  # list of (gamma, beta), len = n_film_points
        film_in = film_list[0]
        film_blocks = film_list[1:]

        ctx: dict = {"B": B, "N": N}
        if n_nodes_per_graph is not None:
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d = Ys.squeeze(0)
            Yc_1d = Yc.squeeze(0)
            (
                undirected, _, edge_index_dir, edge_feat_dir, ys_edge, yc_edge,
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
            ctx["sizes"] = n_nodes_per_graph
        else:
            pairs = self._pairs_for_n(N, device)
            (
                edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list,
            ) = build_edges_plain(
                Line=Line, Ys=Ys, Yc=Yc, N=N,
                edge_feat_dim=self.edge_feat_dim, pairs=pairs, device=device,
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
        ctx["film_in"] = film_in
        ctx["film_blocks"] = film_blocks

        v0 = V0[..., 0]
        th0 = V0[..., 1]
        m0 = v0.new_zeros(B, N, self.d)
        z0 = torch.cat([v0.unsqueeze(-1), th0.unsqueeze(-1), m0], dim=-1)

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
            return out, phys_loss
        return out
