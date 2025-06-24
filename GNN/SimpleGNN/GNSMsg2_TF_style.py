import torch, torch.nn as nn
from torch_scatter import scatter_add          # pip install torch-scatter

def mlp(in_dim, out_dim, hidden, slope=0.01):
    """2-layer Leaky-ReLU MLP (paper spec)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LeakyReLU(slope, inplace=True),
        nn.Linear(hidden, out_dim)
    )

class GNSUndir(nn.Module):
    """
    Graph-Neural Solver – *undirected* edge version
    (no 'from' / 'to' separation).

    d=10 , K=30 , hidden=10 reproduce the paper.
    """
    def __init__(self, d=10, K=30, hidden=10, alpha=1e-3,
                 pinn=True, return_all=False):
        super().__init__()
        self.d, self.K, self.alpha = d, K, alpha
        self.pinn, self.return_all = pinn, return_all

        # single φ for every undirected edge
        self.edge_mlp = nn.ModuleList(
            mlp(2*d + 2, d, hidden) for _ in range(K))

        in_node = 4 + 2*d                       # v,θ,ΔP,ΔQ + m + Σφ
        self.theta_upd = nn.ModuleList(mlp(in_node, 1, hidden) for _ in range(K))
        self.v_upd     = nn.ModuleList(mlp(in_node, 1, hidden) for _ in range(K))
        self.m_upd     = nn.ModuleList(mlp(in_node, d, hidden) for _ in range(K))

        self.decoder   = nn.ModuleList(mlp(d, 1, hidden) for _ in range(K+1))

    # ---------------- helpers -----------------
    @staticmethod
    def calc_PQ(Yr, Yi, v, th):
        V = v * torch.exp(1j*th)
        I = torch.matmul(Yr + 1j*Yi, V.unsqueeze(-1)).squeeze(-1)
        S = V * I.conj()
        return S.real, S.imag

    # build upper-triangle undirected edge list once
    @staticmethod
    def build_edges(mask):
        i, j = torch.triu_indices(*mask.shape, offset=1, device=mask.device)
        keep = mask[i, j].bool()
        return torch.stack([i[keep], j[keep]], 1)      # (E,2)

    # ---------------- forward ------------------
    def forward(self, bus_type, Yr, Yi, P_set, Q_set, V0):
        """
        Shapes are identical to previous class.
        """
        B, N = bus_type.shape
        device = bus_type.device

        with torch.no_grad():
            edge_index = self.build_edges((Yr[0]!=0)|(Yi[0]!=0)).to(device)  # (E,2) why just first sample? --> assume shared in the batch
        E = edge_index.size(0) # number of edges

        G_edge = Yr[:, edge_index[:,0], edge_index[:,1]]   # (B,E)
        B_edge = Yi[:, edge_index[:,0], edge_index[:,1]]   # (B,E)

        v  = V0[...,0].clone();   th = V0[...,1].clone()
        m  = torch.zeros(B, N, self.d, device=device)

        slack = bus_type==1
        pv    = bus_type==2

        phys_loss = torch.zeros(1, device=device)
        outs = []

        for k in range(self.K):
            # 1) mismatches
            P_calc, Q_calc = self.calc_PQ(Yr, Yi, v, th)
            dP = P_set - P_calc;          dQ = Q_set - Q_calc
            dP[slack] = 0;                dQ[slack|pv] = 0

            # 2) messages   φ(m_i , m_j , G_ij , B_ij)
            m_i = m[:, edge_index[:,0]]               # (B,E,d)
            m_j = m[:, edge_index[:,1]]
            edge_feat = torch.stack([G_edge, B_edge], -1)  # (B,E,2)
            φ = self.edge_mlp[k]( torch.cat([m_i, m_j, edge_feat], -1) )

            # scatter to both ends
            idx_i = edge_index[:,0].expand(B, E)
            idx_j = edge_index[:,1].expand(B, E)
            M_neigh = scatter_add(φ, idx_i, dim=1, dim_size=N) + \
                      scatter_add(φ, idx_j, dim=1, dim_size=N)     # (B,N,d)

            # 3) node update
            feats = torch.cat([v.unsqueeze(-1), th.unsqueeze(-1),
                               dP.unsqueeze(-1), dQ.unsqueeze(-1),
                               m, M_neigh], -1)                 # (B,N,4+2d)

            Δθ = self.theta_upd[k](feats).squeeze(-1)
            Δv = self.v_upd[k](feats).squeeze(-1)
            Δm = torch.tanh(self.m_upd[k](feats))

            Δθ[slack] = 0;                   Δv[slack|pv] = 0

            th += Δθ;                       v += Δv;         m += Δm

            if self.return_all:
                outs.append(torch.stack([v, th], -1))

            if self.pinn:
                phys_loss += (0.96**(self.K-1-k))*(dP.pow(2)+dQ.pow(2)).mean()

        V_final = torch.stack([v, th], -1)
        if self.return_all:
            outs.append(V_final)
            return outs, phys_loss if self.pinn else outs
        return (V_final, phys_loss) if self.pinn else V_final