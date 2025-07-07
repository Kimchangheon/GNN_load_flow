# ===== collate_blockdiag.py =====
import torch
from torch.nn.utils.rnn import pad_sequence   # only for vectors
from torch import block_diag                  # torch>=1.10

def collate_blockdiag(samples):
    """
    Turn a list of dict-samples (one grid each) into ONE dict whose
    fields live on a block-diagonal big grid.
    """

    # S_base = torch.cat([s["S_base"] for s in samples], dim=0)      #
    # U_base = torch.cat([s["U_base"] for s in samples], dim=0)      #

    # --- concatenate 1-D fields -----------------------------------------
    bus_type = torch.cat([s["bus_type"] for s in samples], dim=0)      # (ΣN,)
    P_start   = torch.cat([s["P_start"]   for s in samples], dim=0)
    Q_start   = torch.cat([s["Q_start"]   for s in samples], dim=0)
    P_newton   = torch.cat([s["P_newton"]   for s in samples], dim=0)
    Q_newton   = torch.cat([s["Q_newton"]   for s in samples], dim=0)
    V_start   = torch.cat([s["V_start"]   for s in samples], dim=0)      # (ΣN,2)
    V_newton   = torch.cat([s["V_newton"]   for s in samples], dim=0)      # (ΣN,2)

    # --- block-diag 2-D fields -----------------------------------------
    Yr = block_diag(*[s["Ybus_real"] for s in samples])                # (ΣN,ΣN)
    Yi = block_diag(*[s["Ybus_imag"] for s in samples])

    # --- concatenate per-edge fields -----------------------------------
    Lines_connected = torch.cat([s["Lines_connected"] for s in samples], dim=0)
    Y_Lines_real = torch.cat([s["Y_Lines_real"] for s in samples], dim=0)
    Y_Lines_imag = torch.cat([s["Y_Lines_imag"] for s in samples], dim=0)
    Y_C_Lines    = torch.cat([s["Y_C_Lines"]    for s in samples], dim=0)

    # --- bookkeeping: where each grid starts in the long vector --------
    sizes   = [len(s["bus_type"]) for s in samples]                    # list[N_i]
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), 0)[:-1]))


    # ── add the batch dimension (B = 1) ──────────────────────────────
    return {
        "bus_type":  bus_type.unsqueeze(0),   # (1,M)
        "Ybus_real": Yr.unsqueeze(0),         # (1,M,M)
        "Ybus_imag": Yi.unsqueeze(0),
        "P_start":   P_start.unsqueeze(0),  # (1,M)
        "Q_start":   Q_start.unsqueeze(0),
        "P_newton":    P_newton.unsqueeze(0),     # (1,M)
        "Q_newton":    Q_newton.unsqueeze(0),
        "V_start":   V_start.unsqueeze(0),  # (1,M,2)
        "V_newton":    V_newton.unsqueeze(0),     # (1,M,2)

        "Lines_connected": Lines_connected.unsqueeze(0),  # (ΣE, 2)
        "Y_Lines_real": Y_Lines_real.unsqueeze(0),  # (total_edges,)
        "Y_Lines_imag": Y_Lines_imag.unsqueeze(0),
        "Y_C_Lines": Y_C_Lines.unsqueeze(0),
        "offsets":   offsets,                 # keep for optional use
        "sizes":     torch.tensor(sizes)
    }