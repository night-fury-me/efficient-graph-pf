import torch, torch.nn as nn
from torch_scatter import scatter_add
from itertools import combinations          # (or torch.combinations)
import math
import torch.nn.functional as F

# -----------------------------------------------------------------------

# class LearningBlock(nn.Module):  # later change hidden dim to more dims, currently suggested latent=hidden
#     def __init__(self, dim_in, hidden_dim, dim_out):
#         super(LearningBlock, self).__init__()
#         self.linear1 = nn.Linear(dim_in, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear4 = nn.Linear(hidden_dim, dim_out)
#         self.lrelu = nn.LeakyReLU()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.lrelu(x)
#         x = self.linear2(x)
#         x = self.lrelu(x)
#         x = self.linear4(x)
#         return x

class LearningBlock(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.lin1  = nn.Linear(dim_in,  hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)          # NEW
        self.lin2  = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)          # NEW
        self.lin3  = nn.Linear(hidden_dim, dim_out)
        self.act   = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.norm1(self.lin1(x)))
        x = self.act(self.norm2(self.lin2(x)))
        return self.lin3(x)                            # keep last layer raw (or tanh)

class GNSMsg(nn.Module):
    def __init__(self, d:int=32, K:int=30, pinn:bool=True):
        super().__init__()
        self.K, self.d, self.pinn = K, d, pinn

        # edge_in_dim = 2 * d + 3  # m_i , m_j , 3‑line scalars
        edge_in_dim = d + 3  # m_j , 3‑line scalars
        hidden = edge_in_dim
        self.edge_mlp = nn.ModuleList(
            [LearningBlock(edge_in_dim, hidden, d) for _ in range(K)]
        )

        # K *independent* node-update blocks
        in_dim = 4 + d + d            # [v,θ,ΔP,ΔQ] + m_i + Σφ
        hidden = in_dim
        self.theta_upd = nn.ModuleList([LearningBlock(in_dim, hidden, 1) for _ in range(K)])
        self.v_upd     = nn.ModuleList([LearningBlock(in_dim, hidden, 1) for _ in range(K)])
        self.m_upd     = nn.ModuleList([LearningBlock(in_dim, hidden, d) for _ in range(K)])

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph):

        device = bus_type.device
        B, N = bus_type.shape

        Ysr, Ysi = Ys.real, Ys.imag
        P_set, Q_set = S.real, S.imag

        v  = V0[..., 0].clone()        # (B,N) start magnitudes
        θ  = V0[..., 1].clone()        # (B,N) start angles
        m  = torch.zeros(B, N, self.d, device=bus_type.device)

        if n_nodes_per_graph is not None:
            Line = Line.squeeze(0) if Line.dim() == 2 else Line  # 1‑D
            Ysr, Ysi, Yc = Ysr.squeeze(0), Ysi.squeeze(0), Yc.squeeze(0)
            edge_index_parts = []
            edge_feat_parts = []
            deg = []

            ptr = 0  # pointer inside the long Line/Y* vectors
            offset = 0  # node‑index offset for this graph

            for n in n_nodes_per_graph:
                e_all = n * (n - 1) // 2  # how many possible edges
                mask_g = Line[ptr:ptr + e_all]  # slice of length e_all
                ysr_g = Ysr[ptr:ptr + e_all]
                ysi_g = Ysi[ptr:ptr + e_all]
                yc_g = Yc[ptr:ptr + e_all]
                ptr += e_all

                if mask_g.sum() == 0:  # isolated graph – skip everything
                    offset += n
                    deg.append(torch.zeros(n, device=device))
                    continue

                # local indices 0 … n‑1
                pairs_g = torch.tensor(list(combinations(range(n), 2)),
                                       dtype=torch.long, device=device)  # (e_all, 2)

                e_idx_g = pairs_g[mask_g] + offset  # add offset
                edge_index_parts.append(e_idx_g)  # (E_g, 2)

                feat_g = torch.stack([ysr_g[mask_g],
                                      ysi_g[mask_g],
                                      yc_g[mask_g]], dim=-1)  # (E_g, 3)
                edge_feat_parts.append(feat_g)

                # degree for this block
                deg_g = torch.zeros(n, device=device)
                deg_g.index_add_(0, e_idx_g[:, 0] - offset,
                                 torch.ones(e_idx_g.size(0), device=device))
                deg_g.index_add_(0, e_idx_g[:, 1] - offset,
                                 torch.ones(e_idx_g.size(0), device=device))
                deg.append(deg_g)

                offset += n  # next block

            edge_index = torch.cat(edge_index_parts, dim=0)  # (E_total, 2)
            edge_feat = torch.cat(edge_feat_parts, dim=0)  # (E_total, 3)
            deg = torch.cat(deg)  # (N_total,)
            N_total = deg.size(0)
        else :
            pairs = torch.tensor(list(combinations(range(N), 2)),
                                 dtype=torch.long, device=device)  # (28, 2)

            # ----- build edge_index & edge_feat per graph --------------------------
            edge_index_list = []  # list of (E_b, 2) tensors, one per graph
            edge_feat_list = []  # list of (E_b, 3) tensors
            deg = torch.zeros(B, N, device=device)  # node degree

            for b in range(B):
                mask = Line[b]  # (28,) bool
                e = pairs[mask]  # (E_b, 2)
                edge_index_list.append(e)  # store for later

                # pick the three line parameters that correspond to the active edges
                feat_b = torch.stack([Ysr[b, mask],
                                      Ysi[b, mask],
                                      Yc[b, mask]], dim=-1)  # (E_b, 3)

                edge_feat_list.append(feat_b)

                # accumulate degrees for normalisation (undirected graph)
                deg[b].index_add_(0, e[:, 0], torch.ones(e.size(0), device=device))
                deg[b].index_add_(0, e[:, 1], torch.ones(e.size(0), device=device))

        A = deg.clamp_min_(1.).reciprocal()  # (B, N)   1/deg_i
        slack_mask = (bus_type == 1)
        pv_mask    = (bus_type == 2)

        if self.pinn:
            phys_loss = torch.zeros(1, device=A.device)

        for k in range(self.K):
            # 1) power mismatches
            V = v * torch.exp(1j * θ)
            I = torch.matmul(Y, V.unsqueeze(-1)).squeeze(-1)
            S = V * I.conj()
            P_calc, Q_calc = S.real, S.imag
            ΔP = P_set - P_calc
            ΔQ = Q_set - Q_calc
            ΔP[slack_mask] = 0
            ΔQ[slack_mask | pv_mask] = 0      # PV & slack ignore ΔQ

            # ---------------- normalise the four bus scalars --------------------
            bus_feat = torch.stack([v, θ, ΔP, ΔQ], dim=-1)  # (B,N,4)

            M_neigh = torch.zeros(B, N, self.d, device=device)  # will hold Σφ_j→i

            if n_nodes_per_graph is not None:
                # messages
                m_j = m[0 , edge_index[:, 1], :]  # (E_total, d)
                φ_in = torch.cat([m_j, edge_feat], dim=-1)
                φ = self.edge_mlp[k](φ_in)  # (E_total, d)
                agg_i = scatter_add(φ, edge_index[:, 0], dim=0, dim_size=N_total)
                agg_j = scatter_add(φ, edge_index[:, 1], dim=0, dim_size=N_total)
                M_neigh = (agg_i + agg_j) * A.unsqueeze(-1)  # (N_total, d)
                M_neigh = M_neigh.unsqueeze(0)
                # node update exactly as before (v, θ, m all have length N_total now)
            else :
                for b in range(B):  # tiny loop over graphs
                    e_idx = edge_index_list[b]  # (E_b, 2); might be empty
                    if e_idx.numel() == 0:
                        continue  # isolated graph(graph has no edge) – skip

                    # --- messages ------------------------------------------------------
                    # m_i = m[b, e_idx[:, 0], :]  # (E_b, d)
                    m_j = m[b, e_idx[:, 1], :]  # (E_b, d)

                    # φ_in = torch.cat([m_i, m_j, edge_feat_list[b]], dim=-1)  # (E_b, 2d+3)
                    φ_in = torch.cat([m_j, edge_feat_list[b]], dim=-1)  # (E_b, 2d+3)
                    φ = self.edge_mlp[k](φ_in)  # (E_b, d)

                    # --- aggregate Σ_j φ(m_j, line_ij)  -------------------------------
                    agg_i = scatter_add(φ, e_idx[:, 0], dim=0, dim_size=N)  # (N, d)
                    agg_j = scatter_add(φ, e_idx[:, 1], dim=0, dim_size=N)  # (N, d)

                    M_neigh[b] = (agg_i + agg_j) * A[b].unsqueeze(-1)  # degree‑norm

            # 3) node-level update
            # feats = torch.cat([v.unsqueeze(-1), θ.unsqueeze(-1),
            #                    ΔP.unsqueeze(-1), ΔQ.unsqueeze(-1),
            #                    m, M_neigh], dim=-1)        # (B,N,4+2d)
            feats = torch.cat([bus_feat, m, M_neigh], dim=-1)  # (B,N,4+2d) : [v θ ΔP ΔQ] + m_i + Σφ
            Δθ = self.theta_upd[k](feats).squeeze(-1)
            Δv = self.v_upd[k](feats).squeeze(-1)
            Δm = torch.tanh(self.m_upd[k](feats))
            Δm = F.layer_norm(Δm, Δm.shape[-1:])

            # Lock constrained buses (same as before)
            Δθ = Δθ.clone()
            Δv = Δv.clone()
            Δθ[slack_mask] = 0.0  # slack angle fixed
            Δv[slack_mask | pv_mask] = 0.0  # slack & PV magnitude fixed

            v_min, v_max = 0.8, 1.2
            # -------------------- (1) VOLTAGE-STEP LIMITING --------------------
            # Limit |Δθ| <= 0.30 rad; |Δv| <= 10% of current |v|
            dtheta_max = 0.30
            dvm_frac = 0.10
            v_abs = v.abs()  # (B,N)
            Δθ = torch.clamp(Δθ, -dtheta_max, dtheta_max)
            Δv = torch.clamp(Δv, -dvm_frac * v_abs, dvm_frac * v_abs)

            # -------------------- helper: mismatch ∞-norm ----------------------
            def mismatch_inf_norm(v_cand, θ_cand) -> torch.Tensor:
                Vc = v_cand * torch.exp(1j * θ_cand)  # (B,N) complex
                Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)  # (B,N) complex
                Sc = Vc * Ic.conj()
                DP = (P_set - Sc.real).clone()
                DQ = (Q_set - Sc.imag).clone()
                DP[slack_mask] = 0.0
                DQ[slack_mask | pv_mask] = 0.0
                # ∞-norm over buses then over batch
                # (max of max|ΔP| and max|ΔQ| to match power-flow style merit)
                DP_max = DP.abs().amax(dim=-1)  # (B,)
                DQ_max = DQ.abs().amax(dim=-1)  # (B,)
                return torch.maximum(DP_max, DQ_max).amax()  # scalar tensor

            # ---------------- (2) ARMIJO BACKTRACKING LINE-SEARCH --------------
            F0 = mismatch_inf_norm(v, θ)
            alpha = 1.0
            c1 = 1e-4
            shrink = 0.5
            accepted = False

            # Try a small fixed number of backtracks (faster & stable on GPU)
            for _try in range(8):  # α: 1, 0.5, 0.25, ...
                # candidate scaled step
                v_try = torch.clamp(v + alpha * Δv, v_min, v_max)
                θ_try = θ + alpha * Δθ
                θ_try = (θ_try + math.pi) % (2 * math.pi) - math.pi  # wrap to (−π,π]

                F1 = mismatch_inf_norm(v_try, θ_try)

                # Armijo condition: sufficient decrease
                if F1 <= (1.0 - c1 * alpha) * F0:
                    # accept step; also scale Δm consistently
                    v = v_try
                    θ = θ_try
                    m = m + alpha * Δm
                    accepted = True
                    break

                alpha *= shrink

            # If not accepted at all, either keep state or take tiny helpful step
            if not accepted:
                alpha_min = 1e-3
                v_try = torch.clamp(v + alpha_min * Δv, v_min, v_max)
                θ_try = θ + alpha_min * Δθ
                θ_try = (θ_try + math.pi) % (2 * math.pi) - math.pi
                if mismatch_inf_norm(v_try, θ_try) < F0:
                    v, θ, m = v_try, θ_try, m + alpha_min * Δm
                # else: no update (zero step)

            # 5) check for NaNs or Infs
            for name, tensor in {'θ': θ, 'v': v, 'm': m}.items():
                if torch.isnan(tensor).any():
                    print(f"iter {k} NaN detected in {name}")
                if torch.isinf(tensor).any():
                    print(f"iter {k} Inf detected in {name}")

            # ---- physics loss (discounted) ---------------------------------
            if self.pinn:
                step_L = (ΔP**2+ΔQ**2).mean()
                phys_loss = phys_loss + (0.96**(self.K-1-k))*step_L # 0.96**30 = 0.29


        output = torch.stack([v, θ], dim=-1)
        return (output, phys_loss) if self.pinn else output
