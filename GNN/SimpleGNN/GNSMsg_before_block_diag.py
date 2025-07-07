
    def forward(self, bus_type, Line, Yr, Yi, Ysr, Ysi, Yc, P_set, Q_set, V0):

        device = bus_type.device
        B, N = bus_type.shape
        v  = V0[..., 0].clone()        # (B,N) start magnitudes
        θ  = V0[..., 1].clone()        # (B,N) start angles
        m  = torch.zeros(B, N, self.d, device=bus_type.device)

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
        # we can later normalise the aggregated messages by
        # # adjacency (row-normalised)
        # A = ((Yr != 0) | (Yi != 0)).float()
        # idx = torch.arange(N, device=A.device)
        # A[:, idx, idx] = 0 # diagonal = 0
        # A = A / A.sum(-1, keepdim=True).clamp_min(1)

        # # line features  (B,N,N,2)
        # line_feat = torch.stack([Yr, Yi], dim=-1)          # G , B

        # ── edge features  (B,N,N,2)  – z-scored ─────────────────────────

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

            # ---------------- normalise the four bus scalars --------------------
            bus_feat = torch.stack([v, θ, ΔP, ΔQ], dim=-1)  # (B,N,4)
            if self.norm :
                bus_feat = (bus_feat - self.μ_bus) / self.σ_bus

            M_neigh = torch.zeros(B, N, self.d, device=device)  # will hold Σφ_j→i

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
            v = torch.clamp(v, 0.4, 1.2)  # keep magnitude realistic
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