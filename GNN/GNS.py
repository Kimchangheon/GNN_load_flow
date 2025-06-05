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


class Leap(nn.Module):
    """One message-passing block with residual."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, H, A):
        H_neigh = torch.matmul(A, H)          # (N×N)(N×d)→(N×d)
        return H + torch.tanh(self.lin(H_neigh))


class Decoder(nn.Module):
    """d → Δ|V|, Δδ"""
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 2)

    def forward(self, H):
        return self.head(H)                   # (N,2)


class GNSSolver(nn.Module):
    def __init__(self, d=100, K=10):
        super().__init__()
        self.K  = K
        self.emb = BusEmbedding(d)
        self.leap = Leap(d)          # weight shared across K steps
        self.dec  = Decoder(d)

    @staticmethod
    def _build_adjacency(Yr, Yi):
        """dense binary adjacency from real+imag parts"""
        A = ((Yr != 0) | (Yi != 0)).float()
        A.fill_diagonal_(0)
        # row-normalise (optional)
        deg = A.sum(-1, keepdim=True).clamp_min(1.0)
        return A / deg

    def forward(self,
                bus_type,  # (B,N)
                Yr, Yi,  # (B,N,N)
                P_spec, Q_spec,  # (B,N)
                V_start=None):  # (B,N,2) or None
        """
        Returns: V_pred  (B, N, 2)  →  [|V| , δ]
        """
        B, N = bus_type.shape
        device = bus_type.device

        # ------------------------------------------------------------------
        # 1. initialise / flat start: V_start (B,N,2)
        # ------------------------------------------------------------------
        if V_start is None:
            # flat  |V|=1, δ=0
            V_start = torch.zeros(B, N, 2, device=device)
            V_start[..., 0] = 1.0
            # slack magnitude = 1.06
            slack_mask = (bus_type == 1)  # (B,N)
            V_start[..., 0][slack_mask] = 1.06     # ▶ set slack magnitude

        # ------------------------------------------------------------------
        # 2. Build 2-feature vector per bus: feats (B,N,2)
        # ------------------------------------------------------------------
        feats = torch.zeros(B, N, 2, device=device)

        # slack: [ |V| , δ ]
        feats[bus_type == 1] = V_start[bus_type == 1]

        # PV: [ P_spec , |V|_set ]
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
        A = ((Yr != 0) | (Yi != 0)).float()
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = 0.0
        deg = A.sum(-1, keepdim=True).clamp_min(1.0)
        A = A / deg  # (B,N,N)

        # ------------------------------------------------------------------
        # 5. Voltage iteration
        # ------------------------------------------------------------------
        Vmag = V_start[..., 0]  # (B,N)
        Vang = V_start[..., 1]  # (B,N)
        d = H.size(-1)

        for _ in range(self.K):
            # message passing:   H ← H + tanh( A @ H  *W + b )
            H_neigh = torch.matmul(A, H)  # (B,N,d)
            H = H + torch.tanh(self.leap.lin(H_neigh))  # residual

            dV = self.dec.head(H)  # (B,N,2)
            dVm, dVa = dV[..., 0], dV[..., 1]

            # apply bus-type masks
            dVm[bus_type != 3] = 0.  # only PQ adjusts |V|
            dVa[bus_type == 1] = 0.  # slack angle fixed

            # update polar voltages
            Vmag = Vmag + dVm
            Vang = Vang + dVa

        V_pred = torch.stack([Vmag, Vang], dim=-1)  # (B,N,2)
        return V_pred

    # def forward(self, bus_type, Yr, Yi, P_spec, Q_spec,
    #             V_start=None):
    #     """
    #     bus_type : (N,)
    #     Yr,Yi    : (N,N)  real / imag of Y-bus
    #     V_start  : optional (N,2) initial [|V|, δ]  (default flat)
    #     returns  : predicted (N,2)  [|V|, δ]
    #     """
    #     N = bus_type.numel()
    #     device = bus_type.device
    #     if V_start is None:
    #         V_start = torch.cat([torch.ones(N,1,device=device),   # |V|=1 p.u.
    #                              torch.zeros(N,1,device=device)],1)
    #         V_start[bus_type==1,0] = 1.06       # slack |V|
    #     # --- build input features ---------------------------------
    #     # slack:   (|V|,δ) ; PV: (P,|V|set) ; PQ: (P,Q)
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
    #         dVm[bus_type != 3] = 0.0            # only PQ can adjust |V|
    #         dVa[bus_type == 1] = 0.0            # slack angle fixed
    #
    #         # Update polar voltages
    #         Vmag = Vmag + dVm
    #         Vold = Vold + dVa
    #
    #     return torch.stack([Vmag, Vold], dim=1)   # (N,2)