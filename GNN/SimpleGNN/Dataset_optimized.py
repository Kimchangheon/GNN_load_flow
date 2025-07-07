import ast, functools, json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union    # add this

FLOAT_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64

# --------------------------------------------------------------------------- #
#                               helpers                                        #
# --------------------------------------------------------------------------- #
def _safe_list(val) -> list:
    """Always return Python list w/o eval’ing Python code."""
    if isinstance(val, str):
        # 5-10× faster than ast if it’s JSON-style; fallback otherwise
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    return list(val)

def _merge_complex(real, imag, dtype=COMPLEX_DTYPE):
    return np.asarray(real, FLOAT_DTYPE) + 1j * np.asarray(imag, FLOAT_DTYPE)

# --------------------------------------------------------------------------- #
#                              main Dataset                                    #
# --------------------------------------------------------------------------- #
class ChanghunDataset(Dataset):
    """
    Every numeric/list column is eagerly turned into a torch tensor *once* in
    __init__, so __getitem__ is just indexing.
    """
    __slots__ = ("rows", "_per_unit", "_device")

    _LIST_COLS = (
        "bus_typ u_start_real u_start_imag u_newton_real u_newton_imag "
        "Yr Yi Lines_connected Y_Lines_real Y_Lines_imag Y_C_Lines"
    ).split()

    def __init__(self, path: Union[str, Path], * , per_unit: bool = False, device: str = None):
        self._per_unit = per_unit
        self._device   = torch.device(device) if device else None

        # ---------- load & sanitise ------------------------------------------------
        df = pd.read_parquet(path, engine="pyarrow")
        print("Parquet read →", df.shape)

        # • vectorised decode list-columns
        for col in self._LIST_COLS:
            df[col] = df[col].map(_safe_list)

         # • mask diverged rows (variable-length arrays → row-wise test)
        def _is_zero(vec) -> bool:
            arr = np.asarray(vec)
            return arr.size > 0 and np.all(arr == 0)

        keep = ~df.apply(
            lambda r: _is_zero(r.u_newton_real) and _is_zero(r.u_newton_imag),
            axis=1,
        )
        if not keep.all():
            df = df[keep].reset_index(drop=True)
            print(f"Removed {(~keep).sum()} diverged rows → {df.shape}")

        # ---------- convert to tensors --------------------------------------------
        self.rows: List[Dict[str, Any]] = []
        to_t  = functools.partial(torch.as_tensor, device=self._device)

        for _, r in df.iterrows():
            N = int(r.bus_number)

            # scalars
            S_base = r.S_base if per_unit else 1.0
            U_base = r.U_base if per_unit else 1.0
            Y_base = (S_base / U_base ** 2 ) if per_unit else 1.0

            # pre-normalised torch tensors
            row = {
                "S_base": S_base,
                "U_base": U_base,
                "N": N,
                "bus_type": to_t(r.bus_typ, dtype=torch.int64),

                "Ybus_real": to_t(np.array(r.Yr,  FLOAT_DTYPE).reshape(N, N) / Y_base),
                "Ybus_imag": to_t(np.array(r.Yi,  FLOAT_DTYPE).reshape(N, N) / Y_base),

                "Lines_connected": to_t(r.Lines_connected, dtype=torch.bool),
                "Y_Lines_real": to_t(np.array(r.Y_Lines_real, FLOAT_DTYPE) / Y_base),
                "Y_Lines_imag": to_t(np.array(r.Y_Lines_imag, FLOAT_DTYPE) / Y_base),
                "Y_C_Lines":   to_t(np.array(r.Y_C_Lines,   FLOAT_DTYPE) / Y_base),

                # P/Q
                "P_start":  to_t(np.array(r.S_start_real, FLOAT_DTYPE) / S_base),
                "Q_start":  to_t(np.array(r.S_start_imag, FLOAT_DTYPE) / S_base),
                "P_newton": to_t(np.array(r.S_newton_real, FLOAT_DTYPE) / S_base),
                "Q_newton": to_t(np.array(r.S_newton_imag, FLOAT_DTYPE) / S_base),
            }

            # voltages (complex64 → two-channel mag/ang float32)
            U_start  = _merge_complex(r.u_start_real,  r.u_start_imag)  / U_base
            U_newton = _merge_complex(r.u_newton_real, r.u_newton_imag) / U_base
            for name, U in [("start", U_start), ("newton", U_newton)]:
                mag, ang = np.abs(U).astype(FLOAT_DTYPE), np.angle(U).astype(FLOAT_DTYPE)
                row[f"U_{name}"] = to_t(U.astype(COMPLEX_DTYPE))
                row[f"V_{name}"] = to_t(np.stack([mag, ang], axis=1))  # (N,2)

            self.rows.append(row)

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]
