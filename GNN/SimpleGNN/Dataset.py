import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import ast

def _tolist(cell):
    """Make sure we get a Python list even if the parquet cell is a string."""
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return list(cell)

def _merge_complex(real_col, imag_col):
    """Return np.array(real) + 1j*np.array(imag)."""
    real = np.asarray(real_col, dtype=np.float64)
    imag = np.asarray(imag_col,  dtype=np.float64)
    return real + 1j * imag

class ChanghunDataset(Dataset):
    def __init__(self, path):
        df = pd.read_parquet(path)
        self.rec = df.to_dict("records") #small in-memory cache of pure-Python row data, which makes your Dataset implementation both faster and simpler.

    def __len__(self): return len(self.rec)

    def boolvec_to_pairs(self, bool_vec, n_bus):
        """
        Parameters
        ----------
        bool_vec : 1-D Boolean / 0-1 array, length = n_bus*(n_bus-1)//2
        n_bus    : number of buses N

        Returns
        -------
        pairs    : (n_line, 2) int array of [from_bus, to_bus]
        """
        # indices of all upper-triangle pairs in the same order
        i_idx, j_idx = np.triu_indices(n_bus, k=1)  # each is (num_connections,)
        assert bool_vec.size == i_idx.size, "length mismatch with n_bus"

        keep = (bool_vec != 0)  # True where a line exists
        return np.stack([i_idx[keep], j_idx[keep]], axis=1)  # (n_line,2)

    def __getitem__(self, idx):
        r = self.rec[idx]
        N  = int(r["bus_number"])

        # -------- bus types --------
        bus_type = np.array(_tolist(r["bus_typ"]), dtype=np.int64)

        # ----------- line admittances -----------------
        raw_lines = _tolist(r["Lines_connected"])  # 0/1 list
        bool_vec = np.asarray(raw_lines, dtype=bool)
        lines = self.boolvec_to_pairs(bool_vec, N)  # (n_line,2) explicit pairs

        y_series = _merge_complex(r["Y_Lines_real"], r["Y_Lines_imag"])
        y_chg    = 1j * np.asarray(_tolist(r["Y_C_Lines"]), dtype=np.float64)

        Y = np.zeros((N, N), dtype=np.complex128)
        for (f, t), ys, bc in zip(lines, y_series, y_chg):
            Y[f,f] += ys + bc/2
            Y[t,t] += ys + bc/2
            Y[f,t] -= ys
            Y[t,f] -= ys
        Yr, Yi = Y.real.astype(np.float32), Y.imag.astype(np.float32)

        # ------------- strat P,Q -----------------
        S_base = 100 * 1e6
        S_start = _merge_complex(r["S_start_real"], r["S_start_imag"]) / S_base
        P_start = S_start.real.astype(np.float32)
        Q_start = S_start.imag.astype(np.float32)

        # ------------- specified P,Q -----------------
        S_spec = _merge_complex(r["S_newton_real"], r["S_newton_imag"]) / S_base
        P_spec = S_spec.real.astype(np.float32)
        Q_spec = S_spec.imag.astype(np.float32)

        # ------------- start voltages ---------
        U_base = r["U_base"]
        U_start = _merge_complex(r["u_start_real"],
                                r["u_start_imag"]) / U_base
        V_start_mag  = np.abs(U_start).astype(np.float32)
        V_start_ang  = np.angle(U_start).astype(np.float32)
        V_start = np.stack([V_start_mag, V_start_ang], axis=1)   # (N,2)

        # ------------- ground-truth voltages ---------
        U_true = _merge_complex(r["u_newton_real"],
                                r["u_newton_imag"]) / U_base
        V_mag  = np.abs(U_true).astype(np.float32)
        V_ang  = np.angle(U_true).astype(np.float32)
        V_true = np.stack([V_mag, V_ang], axis=1)   # (N,2)

        # -------- convert to torch -------------------
        sample = {
            "bus_type":  torch.from_numpy(bus_type),
            "Ybus_real": torch.from_numpy(Yr),
            "Ybus_imag": torch.from_numpy(Yi),
            "P_start": torch.from_numpy(P_start),
            "Q_start": torch.from_numpy(Q_start),
            "P_spec":    torch.from_numpy(P_spec),
            "Q_spec":    torch.from_numpy(Q_spec),
            "V_start": torch.from_numpy(V_start),
            "V_true":    torch.from_numpy(V_true),
            "N": N  # ← number of buses(expose size for bucketing)

        }
        return sample

