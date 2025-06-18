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
    """
    Graph-Neural Newton solver
    ---------------------------------------------
    • adj_mode ∈ {"default", "mag", "sep", "cplx"}
    • pinn_flag   : use physics-only loss if True
    • K           : # message-passing / Newton steps
    • d           : latent width   (⚠ for 'cplx' use even d)
    ---------------------------------------------
    """
    def __init__(self, pinn_flag=False, adj_mode="default", d=100, K=10):
        super().__init__()
        self.pinn_flag = pinn_flag
        self.adj_mode = adj_mode
        self.K  = K
        self.emb = BusEmbedding(d)
        # ---------- 1. build K *separate* linear blocks -------------
        self.leap_lin = nn.ModuleList()
        for _ in range(K):
            if adj_mode == "sep":               # neighbour tensor (B,N,2d)
                self.leap_lin.append(nn.Linear(2*d+2, d))
            elif adj_mode == "cplx":            # real|imag halves → keep d even
                assert d % 2 == 0, "d must be even for 'cplx'"
                self.leap_lin.append(nn.Linear(d+2, d))
            else:                              # default / mag
                self.leap_lin.append(nn.Linear(d+2, d))
        self.dec  = Decoder(d)

    # ----------------------------------------
    # helper: AC power-flow equations
    # ----------------------------------------
    @staticmethod
    def calc_PQ(Yr, Yi, Vmag, Vang):
        """
        Yr,Yi : (B,N,N) real / imag of Ybus (with diagonals!)
        Vmag  : (B,N)
        Vang  : (B,N)  radians
        returns P,Q  (B,N)
        """
        # build complex voltage vector V = V e^{jθ}
        V = Vmag * torch.exp(1j * Vang)
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
        Vmag = V_start[..., 0]  # (B,N)
        Vang = V_start[..., 1]  # (B,N)
        d = H.size(-1)

        if self.pinn_flag:
            gamma = 0.96  # discount factor  (hyper-param)
            total_loss = torch.zeros(1, device=device)


        for k in range(self.K):
            # ---- compute mismatch loss at this step -------------
            P_calc, Q_calc = self.calc_PQ(Yr, Yi, Vmag, Vang)  # (B,N)
            dP = P_spec - P_calc
            dQ = Q_spec - Q_calc

            #set slack's dP and dQ as 0
            #set PV's dQ as 0
            # indices
            slack_mask = (bus_type == 1)  # (B,N) Booleans
            pv_mask = (bus_type == 2)
            dP[slack_mask] = 0.0  # slack : ignore ∆P
            dQ[slack_mask] = 0.0  # slack : ignore ∆Q
            dQ[pv_mask] = 0.0  # PV    : ignore ∆Q
            mismatch = torch.stack([dP, dQ], dim=-1)  # (B,N,2)

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

            # 2. correction with *iteration-specific* weights ---------
            H_input = torch.cat([H_neigh, mismatch], dim=-1)  # (B,N,d+2)
            H = H + torch.tanh(self.leap_lin[k](H_input))  # residual add

            dV = self.dec.head(H)  # (B,N,2)
            dVm, dVa = dV[..., 0], dV[..., 1]

            # apply bus-type masks
            dVm[bus_type != 3] = 0.  # only PQ adjusts V
            dVa[bus_type == 1] = 0.  # slack angle fixed

            # update polar voltages
            Vmag = Vmag + dVm
            Vang = Vang + dVa

            if self.pinn_flag :
                step_loss = (dP ** 2 + dQ ** 2).mean() * (gamma ** (self.K - 1 - k))
                total_loss = total_loss + step_loss

        V_pred = torch.stack([Vmag, Vang], dim=-1)  # (B,N,2)
        if self.pinn_flag :
            return V_pred, total_loss
        else :
            return V_pred