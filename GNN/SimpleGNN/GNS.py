import torch
import torch.nn as nn
import torch.nn.functional as F

class BusEmbedding(nn.Module):
    """Separate 2→d MLP per bus type."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.slack  = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.gen    = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.load   = nn.Sequential(nn.Linear(2, d), nn.Tanh())

    def forward(self, feat, bus_type):
        """
        feat      : (N,2)  input features  per bus
        bus_type  : (N,)   1|2|3
        """
        out = torch.zeros(feat.size(0), self.d, device=feat.device)
        mask1 = bus_type == 1   # slack
        mask2 = bus_type == 2   # PV
        mask3 = bus_type == 3   # PQ
        out[mask1] = self.slack(feat[mask1])
        out[mask2] = self.gen  (feat[mask2])
        out[mask3] = self.load (feat[mask3])
        return out

class Decoder(nn.Module):
    """d → ΔV, Δδ"""
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 2)

    def forward(self, H):
        return self.head(H)                   # (N,2)


class GNSSolver(nn.Module):
    def __init__(self, pinn_flag=False, adj_mode="default", d=100, K=10):
        super().__init__()
        self.pinn_flag = pinn_flag
        self.adj_mode = adj_mode
        self.K  = K
        self.emb = BusEmbedding(d)
        if adj_mode == "sep":
            self.leap_lin = nn.Linear(2*d, d)
        else : # mag, cplx, default
            if adj_mode == "cplx":
                assert d % 2 == 0, "d must be even for complex mode"
            self.leap_lin = nn.Linear(d, d)
        self.dec  = Decoder(d)

    # ----------------------------------------
    # helper: AC power-flow equations
    # ----------------------------------------
    @staticmethod
    def calc_PQ(Yr, Yi, Vreal, Vang):
        """
        Yr,Yi : (B,N,N) real / imag of Ybus (with diagonals!)
        Vmag  : (B,N)
        Vang  : (B,N)  radians
        returns P,Q  (B,N)
        """
        # build complex voltage vector V = V e^{jθ}
        V = Vreal * torch.exp(1j * Vang)
        I = torch.matmul(Yr + 1j * Yi, V.unsqueeze(-1)).squeeze(-1)  # Y·V
        S = V * I.conj()  # complex power
        return S.real, S.imag  # P,Q

    def forward(self,
                bus_type,  # (B,N)
                Yr, Yi,  # (B,N,N)
                P_spec, Q_spec,  # (B,N)
                V_start=None):  # (B,N,2) or None
        """
        Returns: V_pred  (B, N, 2)  →  [V , δ]
        """
        B, N = bus_type.shape
        device = bus_type.device

        # ------------------------------------------------------------------
        # 1. initialise / flat start: V_start (B,N,2)
        # ------------------------------------------------------------------
        if V_start is None:
            # flat  V=1, δ=0
            V_start = torch.zeros(B, N, 2, device=device)
            V_start[..., 0] = 1.0
            # slack magnitude = 1.06
            slack_mask = (bus_type == 1)  # (B,N)
            V_start[..., 0][slack_mask] = 1.06     # ▶ set slack magnitude

        # ------------------------------------------------------------------
        # 2. Build 2-feature vector per bus: feats (B,N,2)
        # ------------------------------------------------------------------
        feats = torch.zeros(B, N, 2, device=device)

        # slack: [ V , δ ]
        feats[bus_type == 1] = V_start[bus_type == 1]

        # PV: [ P_spec , V_set ]
        pv_mask = (bus_type == 2)
        feats[..., 0][pv_mask] = P_spec[pv_mask]  # correct
        feats[..., 1][pv_mask] = V_start[..., 0][pv_mask]  # correct

        # PQ: [ P_spec , Q_spec ]
        pq_mask = (bus_type == 3)
        feats[..., 0][pq_mask] = P_spec[pq_mask]
        feats[..., 1][pq_mask] = Q_spec[pq_mask]

        # ------------------------------------------------------------------
        # 3. Flatten batch→(BN,2) for embedding, then reshape back
        # ------------------------------------------------------------------
        feats_flat = feats.view(B * N, 2)
        bus_type_flat = bus_type.view(-1)
        H = self.emb(feats_flat, bus_type_flat)  # (BN,d)
        H = H.view(B, N, -1)  # (B,N,d)

        # ------------------------------------------------------------------
        # 4. Build adjacency per sample  A (B,N,N)  row-normalised
        # ------------------------------------------------------------------
        if self.adj_mode == "default": # simple adjacney
            A = ((Yr != 0) | (Yi != 0)).float()
            idx = torch.arange(N, device=A.device)
            A[:, idx, idx] = 0.0
            deg = A.sum(-1, keepdim=True).clamp_min(1.0)
            A = A / deg  # (B,N,N)
        elif self.adj_mode == "mag": # 1. Magnitude-only weight
            # ----- build Y-weighted row-normalised matrix --------------------------
            # Yr,Yi : (B,N,N)
            weight = torch.sqrt(Yr.pow(2) + Yi.pow(2))  # |Y|  (B,N,N)
            weight.diagonal(dim1=-2, dim2=-1).zero_()  # no self-loops

            deg = weight.sum(-1, keepdim=True).clamp_min(1e-6)
            A = weight / deg
        elif self.adj_mode == "sep": # 2. Signed real / imag channels
            G = Yr.clamp(min=0.0)  # take positive part
            B = Yi.abs()
            G.diagonal(dim1=-2, dim2=-1).zero_()
            B.diagonal(dim1=-2, dim2=-1).zero_()

            Ag = G / G.sum(-1, keepdim=True).clamp_min(1e-6)
            Ab = B / B.sum(-1, keepdim=True).clamp_min(1e-6)
        elif self.adj_mode == "cplx":
            d_half = H.size(-1) // 2
            G = Yr                                           # conductance
            B = Yi                                           # susceptance
            G.diagonal(dim1=-2, dim2=-1).zero_()
            B.diagonal(dim1=-2, dim2=-1).zero_()

        # 5. Voltage iteration
        #------------------------------------------------------------------
        Vreal = V_start[..., 0]  # (B,N)
        Vang = V_start[..., 1]  # (B,N)
        d = H.size(-1)

        if self.pinn_flag:
            gamma = 0.9  # discount factor  (hyper-param)
            total_loss = torch.zeros(1, device=device)


        for k in range(self.K):
            # message passing:   H ← H + tanh( A @ H  *W + b )
            if self.adj_mode =="sep" :
                H_g = torch.matmul(Ag, H)
                H_b = torch.matmul(Ab, H)
                H_neigh = torch.cat([H_g, H_b], dim=-1)  # (B,N,2d)
            elif self.adj_mode == "cplx":
                H_r, H_i = H[..., :d_half], H[..., d_half:]
                Hr = torch.matmul(G, H_r) - torch.matmul(B, H_i)
                Hi = torch.matmul(G, H_i) + torch.matmul(B, H_r)
                H_neigh = torch.cat([Hr, Hi], dim=-1)          # (B,N,d)
            else :
                H_neigh = torch.matmul(A, H)  # (B,N,d)
            H = H + torch.tanh(self.leap_lin(H_neigh))  # residual

            dV = self.dec.head(H)  # (B,N,2)
            dVr, dVa = dV[..., 0], dV[..., 1]

            # apply bus-type masks
            dVr[bus_type != 3] = 0.  # only PQ adjusts V
            dVa[bus_type == 1] = 0.  # slack angle fixed

            # update polar voltages
            Vreal = Vreal + dVr
            Vang = Vang + dVa

            if self.pinn_flag :
                # ---- compute mismatch loss at this step -------------
                P_calc, Q_calc = self.calc_PQ(Yr, Yi, Vreal, Vang)  # (B,N)
                dP = P_spec - P_calc
                dQ = Q_spec - Q_calc
                step_loss = (dP ** 2 + dQ ** 2).mean() * (gamma ** (self.K - 1 - k))
                total_loss = total_loss + step_loss

        V_pred = torch.stack([Vreal, Vang], dim=-1)  # (B,N,2)
        if self.pinn_flag :
            return V_pred, total_loss
        else :
            return V_pred

    # def forward(self, bus_type, Yr, Yi, P_spec, Q_spec,
    #             V_start=None):
    #     """
    #     bus_type : (N,)
    #     Yr,Yi    : (N,N)  real / imag of Y-bus
    #     V_start  : optional (N,2) initial [V, δ]  (default flat)
    #     returns  : predicted (N,2)  [V, δ]
    #     """
    #     N = bus_type.numel()
    #     device = bus_type.device
    #     if V_start is None:
    #         V_start = torch.cat([torch.ones(N,1,device=device),   # V=1 p.u.
    #                              torch.zeros(N,1,device=device)],1)
    #         V_start[bus_type==1,0] = 1.06       # slack V
    #     # --- build input features ---------------------------------
    #     # slack:   (V,δ) ; PV: (P,Vset) ; PQ: (P,Q)
    #     feats = torch.zeros(N,2,device=device)
    #     feats[bus_type==1] = V_start[bus_type==1]
    #     feats[bus_type==2,0] = P_spec[bus_type==2]
    #     feats[bus_type==2,1] = V_start[bus_type==2,0]   # voltage set pt
    #     feats[bus_type==3,0] = P_spec[bus_type==3]
    #     feats[bus_type==3,1] = Q_spec[bus_type==3]
    #
    #     H = self.emb(feats, bus_type)           # (N,d)
    #     A = self._build_adjacency(Yr, Yi)       # (N,N)
    #
    #     # working copy of voltage (polar form)
    #     Vmag, Vold = V_start[:,0], V_start[:,1]
    #
    #     for _ in range(self.K):
    #         H = self.leap(H, A)
    #         dV = self.dec(H)                    # (N,2)
    #         dVm, dVa = dV[:,0], dV[:,1]
    #
    #         # Mask per bus type
    #         dVm[bus_type != 3] = 0.0            # only PQ can adjust V
    #         dVa[bus_type == 1] = 0.0            # slack angle fixed
    #
    #         # Update polar voltages
    #         Vmag = Vmag + dVm
    #         Vold = Vold + dVa
    #
    #     return torch.stack([Vmag, Vold], dim=1)   # (N,2)