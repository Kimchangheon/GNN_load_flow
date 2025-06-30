import torch, torch.nn as nn

class LearningBlock(nn.Module):  # later change hidden dim to more dims, currently suggested latent=hidden
    def __init__(self, dim_in, hidden_dim, dim_out):
        super(LearningBlock, self).__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, dim_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        return x

class GNSMsg(nn.Module):
    def __init__(self, d: int = 32, K: int = 30, pinn: bool = True):
        super().__init__()
        self.K, self.d, self.pinn = K, d, pinn
        # φ :  (m_j ,  G_ij , B_ij)  →  d_msg
        self.edge_mlp = nn.Sequential(
            nn.Linear(d + 2, d),
            nn.ReLU(),
            nn.Linear(d, d)
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

    # ---------- forward -------------------------------------------------
    def forward(self, bus_type, Yr, Yi, P_set, Q_set, V0):
        B, N = bus_type.shape
        v  = V0[..., 0].clone()        # (B,N) start magnitudes
        θ  = V0[..., 1].clone()        # (B,N) start angles
        m  = torch.zeros(B, N, self.d, device=bus_type.device)

        # adjacency (row-normalised)
        A = ((Yr != 0) | (Yi != 0)).float()
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = 0
        A = A / A.sum(-1, keepdim=True).clamp_min(1)

        # line features  (B,N,N,2)
        line_feat = torch.stack([Yr, Yi], dim=-1)          # G , B

        slack_mask = (bus_type == 1)
        pv_mask    = (bus_type == 2)

        if self.pinn:
            phys_loss = torch.zeros(1, device=A.device)

        for k in range(self.K):
            # 1) power mismatches
            P_calc, Q_calc = self.calc_PQ(Yr, Yi, v, θ)
            ΔP = P_set - P_calc
            ΔQ = Q_set - Q_calc
            ΔP[slack_mask] = 0
            ΔQ[slack_mask | pv_mask] = 0      # PV & slack ignore ΔQ

            # 2) neighbour messages  Σ_j φ(m_j , G_ij , B_ij) : weighted sum over neighbours.
            m_expanded = m.unsqueeze(2).expand(-1, -1, N, -1)   # (B,N,N,d)
            φ_in  = torch.cat([m_expanded, line_feat], dim=-1)  # (B,N,N,d+2)
            φ     = self.edge_mlp(φ_in)                    # (B,N,N,d)
            # M_neigh = torch.matmul(A, φ)                   # (B,N,d)
            M_neigh = torch.einsum('bij,bijd -> bid', A, φ)  # (B,N,d)
            # M_neigh = (A.unsqueeze(-1) * φ).sum(dim=2)  # (B,N,d) the same with einsum

            # 3) node-level update
            feats = torch.cat([v.unsqueeze(-1), θ.unsqueeze(-1),
                               ΔP.unsqueeze(-1), ΔQ.unsqueeze(-1),
                               m, M_neigh], dim=-1)        # (B,N,4+2d)

            Δθ = self.theta_upd[k](feats).squeeze(-1)
            if torch.isnan(θ).any():  # or torch.isfinite(θ).all()
                print('NaN in θ at iter', k);
                break
            Δv = self.v_upd[k](feats).squeeze(-1)
            Δm = torch.tanh(self.m_upd[k](feats))

            # ---- lock the constrained buses -------------------------
            Δθ[slack_mask] = 0.0          # slack angle fixed
            Δv[slack_mask | pv_mask] = 0  # slack & PV magnitude fixed

            # 4) apply updates
            θ = θ + Δθ
            v = v + Δv
            m = m + Δm

            # ---- physics loss (discounted) ---------------------------------
            if self.pinn:
                step_L = (ΔP**2+ΔQ**2).mean()
                phys_loss = phys_loss + (0.96**(self.K-1-k))*step_L


        output = torch.stack([v, θ], dim=-1)
        return (output, phys_loss) if self.pinn else output