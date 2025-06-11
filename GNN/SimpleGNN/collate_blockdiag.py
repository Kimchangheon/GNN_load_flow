# ===== collate_blockdiag.py =====
import torch
from torch.nn.utils.rnn import pad_sequence   # only for vectors
from torch import block_diag                  # torch>=1.10

def collate_blockdiag(samples):
    """
    Turn a list of dict-samples (one grid each) into ONE dict whose
    fields live on a block-diagonal big grid.
    """
    # --- concatenate 1-D fields -----------------------------------------
    bus_type = torch.cat([s["bus_type"] for s in samples], dim=0)      # (ΣN,)
    P_spec   = torch.cat([s["P_spec"]   for s in samples], dim=0)
    Q_spec   = torch.cat([s["Q_spec"]   for s in samples], dim=0)
    V_true   = torch.cat([s["V_true"]   for s in samples], dim=0)      # (ΣN,2)

    # --- block-diag 2-D fields -----------------------------------------
    Yr = block_diag(*[s["Ybus_real"] for s in samples])                # (ΣN,ΣN)
    Yi = block_diag(*[s["Ybus_imag"] for s in samples])

    # --- bookkeeping: where each grid starts in the long vector --------
    sizes   = [len(s["bus_type"]) for s in samples]                    # list[N_i]
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), 0)[:-1]))

    # ── add the batch dimension (B = 1) ──────────────────────────────
    return {
        "bus_type":  bus_type.unsqueeze(0),   # (1,M)
        "Ybus_real": Yr.unsqueeze(0),         # (1,M,M)
        "Ybus_imag": Yi.unsqueeze(0),
        "P_spec":    P_spec.unsqueeze(0),     # (1,M)
        "Q_spec":    Q_spec.unsqueeze(0),
        "V_true":    V_true.unsqueeze(0),     # (1,M,2)
        "offsets":   offsets,                 # keep for optional use
        "sizes":     torch.tensor(sizes)
    }
    # return {
    #     "bus_type": bus_type,      # (M,)
    #     "Ybus_real": Yr,           # (M,M)
    #     "Ybus_imag": Yi,
    #     "P_spec": P_spec,          # (M,)
    #     "Q_spec": Q_spec,          # (M,)
    #     "V_true": V_true,          # (M,2)
    #     "offsets": offsets,        # (#grids,)  — helps if you need per-grid loss
    #     "sizes":   torch.tensor(sizes)
    # }