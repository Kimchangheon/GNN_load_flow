# Message passing + Bus embedding
# ──────────────────────────────────────────────────────────────────────────────
# Graph-Neural Solver à la PSCC-2020 (minimal version)
# ──────────────────────────────────────────────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F

class BusEmbedding(nn.Module):
    """One small 2→d MLP per bus-type (slack / PV / PQ)."""
    def __init__(self, d):
        super().__init__()
        self.slack = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.gen   = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.load  = nn.Sequential(nn.Linear(2, d), nn.Tanh())

    def forward(self, feat, btype):                       # feat:(BN,2)
        out = torch.zeros(feat.size(0), self.slack[0].out_features,
                          device=feat.device)
        out[btype==1] = self.slack(feat[btype==1])
        out[btype==2] = self.gen  (feat[btype==2])
        out[btype==3] = self.load (feat[btype==3])
        return out                                        # (BN,d)

# ──────────────────────────────────────────────────────────────────────────────
class GNSMsgEmb(nn.Module):
    def __init__(self, d=64, d_msg=32, K=10, pinn=True):
        """
        d      : width of bus-latent hᵢ and memory mᵢ
        d_msg  : width of edge message φ(·)
        K      : # unrolled GN-Newton steps
        pinn   : if True → physics-only loss returned as 2-nd output
        """
        super().__init__()
        self.K, self.pinn = K, pinn
        self.emb = BusEmbedding(d)

        # edge MLP  φ(m_j , line_ij)   (share across iterations & lines)
        self.edge_mlp = nn.Sequential(nn.Linear(d+1, d_msg), nn.Tanh())

        # ─ per-iteration update blocks ───────────────────────────────
        self.L_theta = nn.ModuleList([nn.Linear(d+d_msg+4, 1) for _ in range(K)])
        self.L_v     = nn.ModuleList([nn.Linear(d+d_msg+4, 1) for _ in range(K)])
        self.L_m     = nn.ModuleList([nn.Linear(d+d_msg+4, d) for _ in range(K)])

    # ---------- power-flow helper --------------------------------------------
    @staticmethod
    def _PQ(Yr, Yi, V, theta):
        Vc = V * torch.exp(1j*theta)                       # complex voltage
        I  = torch.matmul(Yr+1j*Yi, Vc.unsqueeze(-1)).squeeze(-1)
        S  = Vc * I.conj()
        return S.real, S.imag                             # (B,N)

    # ------------------------------------------------------------------------
    def forward(self, btype, Yr, Yi, P_set, Q_set, V0=None):
        """
        btype : (B,N)   1(slack)|2(PV)|3(PQ)
        Yr/Yi : (B,N,N) Y-bus
        P_set/Q_set : specified injections  (B,N)
        V0    : optional flat-start  (B,N,2) → [|V| , θ]
        returns V_pred (and physics loss if self.pinn)
        """
        B,N = btype.shape ; dev=btype.device
        if V0 is None:                                    # flat start
            V0 = torch.zeros(B,N,2, device=dev)
            V0[...,0] = 1.0
            V0[btype==1,0] = 1.06

        V, theta = V0[...,0], V0[...,1]                  # (B,N)

        # ---------- initial embeddings & memory -------------------------------
        feat0 = torch.stack([P_set, Q_set], -1).view(-1,2)    # (BN,2)
        H = self.emb(feat0, btype.view(-1)).view(B,N,-1)      # (B,N,d)
        m = torch.zeros_like(H)                               # memory

        phys_loss = 0.0

        # ---------- constant adjacency mask (binary) --------------------------
        A = ((Yr!=0)|(Yi!=0)).float() ; A.diagonal(dim1=-2,dim2=-1).zero_()

        for k in range(self.K):
            # ---- 1. power mismatches ----------------------------------------
            P_cal, Q_cal = self._PQ(Yr, Yi, V, theta)
            dP, dQ = P_set-P_cal, Q_set-Q_cal

            # ignore ∆Q at PV, ignore both at slack
            dP = dP.masked_fill(btype==1, 0.)
            dQ = dQ.masked_fill(btype<=2, 0.)

            # ---- 2. edge messages  φ(m_j , |y_ij|) --------------------------
            # simple scalar line-feature: |Y|
            Ymag = torch.sqrt(Yr**2+Yi**2).unsqueeze(-1)      # (B,N,N,1)
            msg  = self.edge_mlp( torch.cat([
                        m.unsqueeze(2).expand(-1,-1,N,-1),    # m_j
                        Ymag                                   # |y_ij|
                    ], -1) )                                  # (B,N,N,d_msg)
            M_neigh = torch.matmul(A, msg)                    # Σ_j φ → (B,N,d_msg)

            # ---- 3. local feature vector  z_i -------------------------------
            z = torch.cat([ V.unsqueeze(-1), theta.unsqueeze(-1),
                            dP.unsqueeze(-1), dQ.unsqueeze(-1),
                            m, M_neigh ], -1)                 # (B,N, d+d_msg+4)

            # ---- 4. predictor heads ----------------------------------------
            d_theta = self.L_theta[k](z).squeeze(-1)          # (B,N)
            d_v     = self.L_v    [k](z).squeeze(-1)          # (B,N)
            d_m     = self.L_m    [k](z)                      # (B,N,d)

            # bus-type masks
            d_theta = d_theta.masked_fill(btype==1, 0.)       # slack θ fixed
            d_v     = d_v.masked_fill(btype!=3, 0.)           # only PQ |V|

            # ---- 5. updates -------------------------------------------------
            theta = theta + d_theta
            V     = V     + d_v
            m     = m     + d_m

            # ---- physics loss (discounted) ---------------------------------
            if self.pinn:
                step_L = (dP**2+dQ**2).mean()
                phys_loss = phys_loss + (0.96**(self.K-1-k))*step_L

        V_pred = torch.stack([V, theta], -1)                  # (B,N,2)
        return (V_pred, phys_loss) if self.pinn else V_pred