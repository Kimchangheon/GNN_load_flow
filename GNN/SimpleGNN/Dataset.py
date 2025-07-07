import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import ast

FLOAT_DTYPE = np.float64
COMPLEX_DTYPE = np.complex128

def _tolist(cell):
    """Make sure we get a Python list even if the parquet cell is a string."""
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return list(cell)

def _merge_complex(real_col, imag_col):
    """Return np.array(real) + 1j*np.array(imag)."""
    real = np.asarray(real_col, dtype=FLOAT_DTYPE)
    imag = np.asarray(imag_col,  dtype=FLOAT_DTYPE)
    return real + 1j * imag

def row_has_zero_voltage(row) -> bool:
    """True if *both* u_newton columns in this row are lists / arrays of *all* zeros."""
    real_vals = _tolist(row["u_newton_real"])
    imag_vals = _tolist(row["u_newton_imag"])
    return real_vals and imag_vals and all(v == 0 for v in real_vals + imag_vals)


class ChanghunDataset(Dataset):
    def __init__(self, path, per_unit=False, device=None):
        self.per_unit = per_unit
        df = pd.read_parquet(path)
        print(f"Original shape: {df.shape}")

        # ------------------------ drop diverged Newton rows ---------------------
        mask_diverged = df.apply(row_has_zero_voltage, axis=1)
        n_removed = mask_diverged.sum()
        if n_removed:
            df = df[~mask_diverged].reset_index(drop=True)
            print(f"Removed {n_removed} diverged rows → new shape: {df.shape}")
        else:
            print("No diverged rows found")

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

        # ------------- strat P,Q -----------------

        S_base = r["S_base"] if self.per_unit else 1.0
        P_start = r["S_start_real"] / S_base
        Q_start = r["S_start_imag"] / S_base
        S_start = P_start + 1j * Q_start

        # ------------- specified P,Q -----------------
        P_newton = r["S_newton_real"] / S_base
        Q_newton = r["S_newton_imag"] / S_base
        S_newton = P_newton + 1j * Q_newton

        # ------------- start voltages ---------
        U_base = r["U_base"] if self.per_unit else 1.0
        U_start = r["u_start_real"] + 1j * r["u_start_imag"]
        U_start = U_start / U_base
        # U_start = _merge_complex(r["u_start_real"],
        #                         r["u_start_imag"]) / U_base
        V_start_mag  = np.abs(U_start)
        V_start_ang  = np.angle(U_start)
        V_start = np.stack([V_start_mag, V_start_ang], axis=1)   # (N,2)
        U_recovered = V_start_mag * np.exp(1j * V_start_ang)
        # ------------- ground-truth voltages ---------
        U_newton = _merge_complex(r["u_newton_real"],
                                r["u_newton_imag"]) / U_base
        V_newton_mag  = np.abs(U_newton)
        V_newton_ang  = np.angle(U_newton)
        V_newton = np.stack([V_newton_mag, V_newton_ang], axis=1)   # (N,2)

        # ----------- line admittances -----------------
        # raw_lines = _tolist(r["Lines_connected"])  # 0/1 list
        # bool_vec = np.asarray(raw_lines, dtype=bool)
        # lines = self.boolvec_to_pairs(bool_vec, N)  # (n_line,2) explicit pairs
        #
        # Z_Base = U_base ** 2 / (S_base)
        #
        # y_series = _merge_complex(r["Y_Lines_real"], r["Y_Lines_imag"]) * Z_Base
        # y_chg    = 1j * np.asarray(_tolist(r["Y_C_Lines"]), dtype=np.float64) / Z_Base
        #
        # Y = np.zeros((N, N), dtype=np.complex128)
        # for (f, t), ys, bc in zip(lines, y_series, y_chg):
        #     Y[f,f] += ys + bc/2
        #     Y[t,t] += ys + bc/2
        #     Y[f,t] -= ys
        #     Y[t,f] -= ys
        # Yr, Yi = Y.real.astype(np.float32), Y.imag.astype(np.float32)

        # def create_adjacency_matrix(bus_number, lines_connected):
        #     # Create an adjacency matrix to represent connections
        #     adjacency_matrix = np.zeros((bus_number, bus_number), dtype=int)
        #
        #     # Fill the adjacency matrix based on the lines_connected array
        #     connection_index = 0
        #     for i in range(bus_number):
        #         for j in range(i + 1, bus_number):
        #             if lines_connected[connection_index] == 1:
        #                 adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1
        #             connection_index += 1
        #
        #     return adjacency_matrix
        #
        # def insert_values_in_matrix(matrix, connections, values):
        #     # Create a copy of the matrix to avoid modifying the original
        #     matrix_with_values = np.array(matrix, dtype=float)
        #     new_values = connections * values
        #     value_index = 0
        #     for i in range(matrix.shape[0]):
        #         for j in range(i + 1, matrix.shape[1]):
        #             if matrix[i][j] == 1:
        #                 # Insert the value regardless of the connection array (since the matrix itself defines the connections)
        #                 matrix_with_values[i][j] = matrix_with_values[j][i] = values[value_index]
        #             value_index += 1
        #
        #     return matrix_with_values
        #
        # def insert_values_in_matrix_komplex(matrix, connections, values):
        #     # Create a copy of the matrix to avoid modifying the original
        #     matrix_with_values = np.array(matrix, dtype=complex)
        #     new_values = connections * values
        #     value_index = 0
        #     for i in range(matrix.shape[0]):
        #         for j in range(i + 1, matrix.shape[1]):
        #             if matrix[i][j] == 1:
        #                 # Insert the complex value regardless of the connection array (since the matrix itself defines the connections)
        #                 matrix_with_values[i][j] = matrix_with_values[j][i] = values[value_index]
        #             value_index += 1
        #
        #     return matrix_with_values
        #
        # def build_Y_matrix(matrix, Y_C_Bus, Line_matrix):
        #     # Create a copy of the matrix to avoid modifying the original
        #     Y_matrix = np.zeros_like(matrix, dtype=complex)
        #     value_index = 0
        #     for i in range(matrix.shape[0]):
        #         for j in range(matrix.shape[1]):
        #             if i == j:
        #                 Y_matrix[i][i] = Y_matrix[i][i] + 1j * Y_C_Bus[i]
        #             else:
        #                 if matrix[i][j] == 1:
        #                     Y_matrix[i][j] = Y_matrix[j][i] = -Line_matrix[i][j]
        #                     Y_matrix[i][i] = Y_matrix[i][i] + Line_matrix[i][j]
        #             value_index += 1
        #
        #     return Y_matrix
        #
        # Y_Lines = _merge_complex(r["Y_Lines_real"], r["Y_Lines_imag"])
        # Y_C_Lines = np.asarray(_tolist(r["Y_C_Lines"]), dtype=np.float64)
        # Lines_connected = _tolist(r["Lines_connected"])
        # Conection_matrix = create_adjacency_matrix(N, Lines_connected)
        #
        # Y_C_Bus = np.sum(insert_values_in_matrix(Conection_matrix, Lines_connected, Y_C_Lines), axis=0)
        #
        # Line_matrix = insert_values_in_matrix_komplex(Conection_matrix, Lines_connected, Y_Lines)
        # Y = build_Y_matrix(Conection_matrix, Y_C_Bus, Line_matrix)
        # Yr, Yi = Y.real.astype(np.float32), Y.imag.astype(np.float32)




        # -------- convert to torch -------------------

        Z_base = U_base ** 2 / (S_base)  # 121
        Y_Base = 1 / Z_base if self.per_unit else 1.0

        Yr = r["Yr"].copy().reshape((N, N)).astype(FLOAT_DTYPE) / Y_Base
        Yi = r["Yi"].copy().reshape((N, N)).astype(FLOAT_DTYPE) / Y_Base

        # ----------- line admittances -----------------
        # Lines_connected = r["Lines_connected"].copy()
        # Lines_connected = _tolist(r["Lines_connected"])  # 0/1 list
        # bool_vec = np.asarray(raw_lines, dtype=bool)
        # lines_from_to = self.boolvec_to_pairs(bool_vec, N)  # (n_line,2) explicit pairs # from to

        Y_Lines_real = r["Y_Lines_real"].copy().astype(FLOAT_DTYPE) / Y_Base
        Y_Lines_imag = r["Y_Lines_imag"].copy().astype(FLOAT_DTYPE) / Y_Base
        Y_C_Lines = r["Y_C_Lines"].copy().astype(FLOAT_DTYPE) / Y_Base

        sample = {
            "N": N,  # ← number of buses(expose size for bucketing)
            "bus_type":  torch.from_numpy(bus_type),
            "Ybus_real": torch.from_numpy(Yr),
            "Ybus_imag": torch.from_numpy(Yi),

            "Lines_connected" : torch.tensor(r["Lines_connected"], dtype=torch.bool),
            "Y_Lines_real": torch.from_numpy(Y_Lines_real),
            "Y_Lines_imag": torch.from_numpy(Y_Lines_imag),
            "Y_C_Lines": torch.from_numpy(Y_C_Lines),

            "P_start": torch.from_numpy(P_start),
            "Q_start": torch.from_numpy(Q_start),
            "P_newton":    torch.from_numpy(P_newton),
            "Q_newton":    torch.from_numpy(Q_newton),

            "U_start": torch.from_numpy(U_start),
            "V_start": torch.from_numpy(V_start),
            "U_newton": torch.from_numpy(U_newton),
            "V_newton":    torch.from_numpy(V_newton)
        }
        return sample

