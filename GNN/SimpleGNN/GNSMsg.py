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
    def __init__(self, d:int=32, K:int=30, pinn:bool=True,
                 norm=False, μ_bus=None, σ_bus=None, μ_edge=None, σ_edge=None):
        super().__init__()
        # ---- store normalisation params --------------------------------- adds tensor to the module as a persistent, non-trainable member:
        self.norm = norm
        if self.norm:
            self.register_buffer('μ_bus',  μ_bus.view(1,1,4))   # shape (1,1,4)
            self.register_buffer('σ_bus',  σ_bus.view(1,1,4))
            self.register_buffer('μ_edge', μ_edge.view(1,1,1,2))# shape (1,1,1,2)
            self.register_buffer('σ_edge', σ_edge.view(1,1,1,2))

        self.K, self.d, self.pinn = K, d, pinn
        # φ :  (m_j ,  G_ij , B_ij)  →  d_msg
        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(d + 2, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d)
        # )

        # edge_in_dim = 2 * d + 3  # m_i , m_j , 3‑line scalars
        edge_in_dim = d + 3  # m_j , 3‑line scalars
        hidden = edge_in_dim
        self.edge_mlp = nn.ModuleList(
            [LearningBlock(edge_in_dim, hidden, d) for _ in range(K)]
        )

        # K *independent* node-update blocks
        in_dim = 4 + d + d            # [v,θ,ΔP,ΔQ] + m_i + Σφ
        hidden = in_dim
        # self.theta_upd = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(K)])
        # self.v_upd     = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(K)])
        # self.m_upd     = nn.ModuleList([nn.Linear(in_dim, d) for _ in range(K)])
        self.theta_upd = nn.ModuleList([LearningBlock(in_dim, hidden, 1) for _ in range(K)])
        self.v_upd     = nn.ModuleList([LearningBlock(in_dim, hidden, 1) for _ in range(K)])
        self.m_upd     = nn.ModuleList([LearningBlock(in_dim, hidden, d) for _ in range(K)])

    # ---------- helper --------------------------------------------------
    @staticmethod
    def calc_PQ(Yr, Yi, v, θ):
        V = v * torch.exp(1j * θ)
        I = torch.matmul((Yr + 1j * Yi), V.unsqueeze(-1)).squeeze(-1)
        S = V * I.conj()
        return S.real, S.imag

    # ------------------------------------------------------------------

    # def forward(self, bus_type, Line, Yr, Yi, Ysr, Ysi, Yc, P_set, Q_set, V0, n_nodes_per_graph):
    #
    #     device = bus_type.device
    #     B, N = bus_type.shape
    #     v  = V0[..., 0].clone()        # (B,N) start magnitudes
    #     θ  = V0[..., 1].clone()        # (B,N) start angles
    #     m  = torch.zeros(B, N, self.d, device=bus_type.device)
    #
    #     if n_nodes_per_graph is None:
    #         pairs = torch.tensor(list(combinations(range(N), 2)),
    #                              dtype=torch.long, device=device)  # (28, 2)
    #
    #         # ----- build edge_index & edge_feat per graph --------------------------
    #         edge_index_list = []  # list of (E_b, 2) tensors, one per graph
    #         edge_feat_list = []  # list of (E_b, 3) tensors
    #         deg = torch.zeros(B, N, device=device)  # node degree
    #         print("N shape ", N)
    #         print("deg shape ", deg.shape)
    #         for b in range(B):
    #             mask = Line[b]  # (28,) bool
    #             e = pairs[mask]  # (E_b, 2)
    #             edge_index_list.append(e)  # store for later
    #
    #             # pick the three line parameters that correspond to the active edges
    #             feat_b = torch.stack([Ysr[b, mask],
    #                                   Ysi[b, mask],
    #                                   Yc[b, mask]], dim=-1)  # (E_b, 3)
    #
    #             if self.norm:
    #                 feat_b = (feat_b - self.μ_edge.squeeze(0)) / self.σ_edge.squeeze(0)
    #             edge_feat_list.append(feat_b)
    #
    #             # accumulate degrees for normalisation (undirected graph)
    #             deg[b].index_add_(0, e[:, 0], torch.ones(e.size(0), device=device))
    #             deg[b].index_add_(0, e[:, 1], torch.ones(e.size(0), device=device))
    #     else :
    #         Line = Line.squeeze(0) if Line.dim() == 2 else Line  # 1‑D
    #         Ysr, Ysi, Yc = Ysr.squeeze(0), Ysi.squeeze(0), Yc.squeeze(0)
    #         edge_index_parts = []
    #         edge_feat_parts = []
    #         deg = []
    #
    #         ptr = 0  # pointer inside the long Line/Y* vectors
    #         offset = 0  # node‑index offset for this graph
    #
    #         for n in n_nodes_per_graph:
    #             e_all = n * (n - 1) // 2  # how many possible edges
    #             mask_g = Line[ptr:ptr + e_all]  # slice of length e_all
    #             ysr_g = Ysr[ptr:ptr + e_all]
    #             ysi_g = Ysi[ptr:ptr + e_all]
    #             yc_g = Yc[ptr:ptr + e_all]
    #             ptr += e_all
    #
    #             if mask_g.sum() == 0:  # isolated graph – skip everything
    #                 offset += n
    #                 deg.append(torch.zeros(n, device=device))
    #                 continue
    #
    #             # local indices 0 … n‑1
    #             pairs_g = torch.tensor(list(combinations(range(n), 2)),
    #                                    dtype=torch.long, device=device)  # (e_all, 2)
    #
    #             e_idx_g = pairs_g[mask_g] + offset  # add offset
    #             edge_index_parts.append(e_idx_g)  # (E_g, 2)
    #
    #             feat_g = torch.stack([ysr_g[mask_g],
    #                                   ysi_g[mask_g],
    #                                   yc_g[mask_g]], dim=-1)  # (E_g, 3)
    #             edge_feat_parts.append(feat_g)
    #
    #             # degree for this block
    #             deg_g = torch.zeros(n, device=device)
    #             deg_g.index_add_(0, e_idx_g[:, 0] - offset,
    #                              torch.ones(e_idx_g.size(0), device=device))
    #             deg_g.index_add_(0, e_idx_g[:, 1] - offset,
    #                              torch.ones(e_idx_g.size(0), device=device))
    #             deg.append(deg_g)
    #
    #             offset += n  # next block
    #
    #         edge_index = torch.cat(edge_index_parts, dim=0)  # (E_total, 2)
    #         edge_feat = torch.cat(edge_feat_parts, dim=0)  # (E_total, 3)
    #         deg = torch.cat(deg)  # (N_total,)
    #
    #     N = deg.size(0)
    #     A = deg.clamp_min_(1.).reciprocal()  # (B, N)   1/deg_i
    #     print("deg shape ", deg.shape)
    #     print("A shape ", A.shape)
    #
    #
    #     slack_mask = (bus_type == 1)
    #     pv_mask    = (bus_type == 2)
    #
    #     if self.pinn:
    #         phys_loss = torch.zeros(1, device=A.device)
    #
    #     for k in range(self.K):
    #         # 1) power mismatches
    #         P_calc, Q_calc = self.calc_PQ(Yr, Yi, v, θ)
    #         ΔP = P_set - P_calc
    #         ΔQ = Q_set - Q_calc
    #         ΔP[slack_mask] = 0
    #         ΔQ[slack_mask | pv_mask] = 0      # PV & slack ignore ΔQ
    #
    #         # ---------------- normalise the four bus scalars --------------------
    #         bus_feat = torch.stack([v, θ, ΔP, ΔQ], dim=-1)  # (B,N,4)
    #         if self.norm :
    #             bus_feat = (bus_feat - self.μ_bus) / self.σ_bus
    #
    #         M_neigh = torch.zeros(B, N, self.d, device=device)  # will hold Σφ_j→i
    #
    #         if n_nodes_per_graph is not None:
    #             # messages
    #             m_j = m[0 , edge_index[:, 1], :]  # (E_total, d)
    #             φ_in = torch.cat([m_j, edge_feat], dim=-1)
    #             φ = self.edge_mlp[k](φ_in)  # (E_total, d)
    #             N_total = N
    #             agg_i = scatter_add(φ, edge_index[:, 0], dim=0, dim_size=N_total)
    #             agg_j = scatter_add(φ, edge_index[:, 1], dim=0, dim_size=N_total)
    #             M_neigh = (agg_i + agg_j) * A.unsqueeze(-1)  # (N_total, d)
    #             M_neigh = M_neigh.unsqueeze(0)
    #             # node update exactly as before (v, θ, m all have length N_total now)
    #         else :
    #             for b in range(B):  # tiny loop over graphs
    #                 e_idx = edge_index_list[b]  # (E_b, 2); might be empty
    #                 if e_idx.numel() == 0:
    #                     continue  # isolated graph(graph has no edge) – skip
    #
    #                 # --- messages ------------------------------------------------------
    #                 # m_i = m[b, e_idx[:, 0], :]  # (E_b, d)
    #                 m_j = m[b, e_idx[:, 1], :]  # (E_b, d)
    #
    #                 # φ_in = torch.cat([m_i, m_j, edge_feat_list[b]], dim=-1)  # (E_b, 2d+3)
    #                 φ_in = torch.cat([m_j, edge_feat_list[b]], dim=-1)  # (E_b, 2d+3)
    #                 φ = self.edge_mlp[k](φ_in)  # (E_b, d)
    #
    #                 # --- aggregate Σ_j φ(m_j, line_ij)  -------------------------------
    #                 agg_i = scatter_add(φ, e_idx[:, 0], dim=0, dim_size=N)  # (N, d)
    #                 agg_j = scatter_add(φ, e_idx[:, 1], dim=0, dim_size=N)  # (N, d)
    #
    #                 M_neigh[b] = (agg_i + agg_j) * A[b].unsqueeze(-1)  # degree‑norm
    #
    #         # 3) node-level update
    #         # feats = torch.cat([v.unsqueeze(-1), θ.unsqueeze(-1),
    #         #                    ΔP.unsqueeze(-1), ΔQ.unsqueeze(-1),
    #         #                    m, M_neigh], dim=-1)        # (B,N,4+2d)
    #         feats = torch.cat([bus_feat, m, M_neigh], dim=-1)  # (B,N,4+2d)
    #
    #         Δθ = self.theta_upd[k](feats).squeeze(-1)
    #         Δv = self.v_upd[k](feats).squeeze(-1)
    #         Δm = torch.tanh(self.m_upd[k](feats))
    #         Δm = F.layer_norm(Δm, Δm.shape[-1:])  # quick 1-liner if you don’t want a module
    #         # ---- lock the constrained buses -------------------------
    #         Δθ[slack_mask] = 0.0          # slack angle fixed
    #         Δv[slack_mask | pv_mask] = 0  # slack & PV magnitude fixed
    #
    #         # 4) apply updates
    #         θ = θ + Δθ
    #         v = v + Δv
    #         m = m + Δm
    #
    #         # after you update v and θ
    #         v = torch.clamp(v, 0.4, 1.2)  # keep magnitude realistic
    #         θ = (θ + math.pi) % (2 * math.pi) - math.pi  # wrap → (–π, π]
    #
    #         # 5) check for NaNs or Infs
    #         for name, tensor in {'θ': θ, 'v': v, 'm': m}.items():
    #             if torch.isnan(tensor).any():
    #                 print(f"iter {k} NaN detected in {name}")
    #             if torch.isinf(tensor).any():
    #                 print(f"iter {k} Inf detected in {name}")
    #
    #         # ---- physics loss (discounted) ---------------------------------
    #         if self.pinn:
    #             step_L = (ΔP**2+ΔQ**2).mean()
    #             phys_loss = phys_loss + (0.96**(self.K-1-k))*step_L
    #
    #
    #     output = torch.stack([v, θ], dim=-1)
    #     return (output, phys_loss) if self.pinn else output


    def forward(self, bus_type, Line, Yr, Yi, Ysr, Ysi, Yc, P_set, Q_set, V0, Ustart, n_nodes_per_graph):

        device = bus_type.device
        B, N = bus_type.shape
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

                if self.norm:
                    feat_b = (feat_b - self.μ_edge.squeeze(0)) / self.σ_edge.squeeze(0)
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
            I = torch.matmul((Yr + 1j * Yi), V.unsqueeze(-1)).squeeze(-1)
            S = V * I.conj()
            P_calc, Q_calc = S.real, S.imag
            # P_calc, Q_calc = self.calc_PQ(Yr, Yi, v, θ)
            ΔP = P_set - P_calc
            ΔQ = Q_set - Q_calc
            ΔP[slack_mask] = 0
            ΔQ[slack_mask | pv_mask] = 0      # PV & slack ignore ΔQ

            # ---------------- normalise the four bus scalars --------------------
            bus_feat = torch.stack([v, θ, ΔP, ΔQ], dim=-1)  # (B,N,4)
            if self.norm :
                bus_feat = (bus_feat - self.μ_bus) / self.σ_bus

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
            feats = torch.cat([bus_feat, m, M_neigh], dim=-1)  # (B,N,4+2d)

            Δθ = self.theta_upd[k](feats).squeeze(-1)
            Δv = self.v_upd[k](feats).squeeze(-1)
            Δm = torch.tanh(self.m_upd[k](feats))
            Δm = F.layer_norm(Δm, Δm.shape[-1:])  # quick 1-liner if you don’t want a module
            # ---- lock the constrained buses -------------------------
            Δθ[slack_mask] = 0.0          # slack angle fixed
            Δv[slack_mask | pv_mask] = 0  # slack & PV magnitude fixed

            # 4) apply updates
            θ = θ + Δθ
            v = v + Δv
            m = m + Δm

            # after you update v and θ
            v = torch.clamp(v, 0.4, 1.2) # keep magnitude realistic
            θ = (θ + math.pi) % (2 * math.pi) - math.pi  # wrap → (–π, π]

            # 5) check for NaNs or Infs
            for name, tensor in {'θ': θ, 'v': v, 'm': m}.items():
                if torch.isnan(tensor).any():
                    print(f"iter {k} NaN detected in {name}")
                if torch.isinf(tensor).any():
                    print(f"iter {k} Inf detected in {name}")

            # ---- physics loss (discounted) ---------------------------------
            if self.pinn:
                step_L = (ΔP**2+ΔQ**2).mean()
                phys_loss = phys_loss + (0.96**(self.K-1-k))*step_L


        output = torch.stack([v, θ], dim=-1)
        return (output, phys_loss) if self.pinn else output