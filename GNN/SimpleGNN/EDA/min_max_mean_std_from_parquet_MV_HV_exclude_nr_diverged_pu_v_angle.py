#!/usr/bin/env python3
"""
stats_pu.py – column‑wise statistics **plus per‑unit values**
===========================================================

* Drops diverged Newton–Raphson rows (where **both** `u_newton_real` and
  `u_newton_imag` are all‑zero).
* Reads **`U_base`** and **`S_base`** *from the Parquet columns themselves* – no
  command‑line flags needed.
* Computes the electrical bases

      Z_base = U_base ** 2 / S_base
      Y_base = 1 / Z_base

  for each row (works even if the system uses multiple base sets).
* Prints per‑column **min / max / mean / std** in **physical units** *and* in
  **per‑unit (p.u.)** when a sensible base exists:

  | Column name prefix | Per‑unit divisor |
  |--------------------|------------------|
  | `Y_`               | `Y_base`         |
  | `P_`, `Q_`, `S_`   | `S_base`         |
  | `V_`, `U_`, `u_`   | `U_base`         |
* Reports **global min / max** of

  * voltage magnitude **|V|**          (physical & p.u.)
  * voltage angle      **θ [rad]**    (physical only – p.u. angle == physical)

  derived from the complex vector `u = u_real + 1j*u_imag` in each row.

Usage
-----
    python stats_pu.py path/to/file.parquet
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

###############################################################################
# helpers
###############################################################################

_NUMBER_RE = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$")


def _tolist(cell) -> list[float]:
    """Return *flat* list of floats contained in *cell* (scalar, list‑str, etc.)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    if isinstance(cell, (int, float, np.number)):
        return [cell]
    if isinstance(cell, str):
        s = cell.strip()
        if s and s[0] in "[(":
            try:
                return list(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                pass
        if _NUMBER_RE.match(s):
            try:
                return [float(s)]
            except ValueError:
                pass
        return []
    raise TypeError(f"Unsupported cell type: {type(cell)}")


def row_has_zero_voltage(row) -> bool:
    """True if *both* u_newton columns are arrays of **all zeros**."""
    real_vals = _tolist(row["u_newton_real"])
    imag_vals = _tolist(row["u_newton_imag"])
    return real_vals and imag_vals and all(v == 0 for v in real_vals + imag_vals)


def infer_divisor(col: str, u_base: float, s_base: float, y_base: float) -> float | None:
    """Return appropriate base for *col* or None if not applicable."""
    if col.startswith("Y_"):
        return y_base
    if col.startswith(("P_", "Q_", "S_")):
        return s_base
    if col.startswith(("V_", "U_", "u_")):
        return u_base
    return None

###############################################################################
# main
###############################################################################

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="Column‑wise stats (+p.u.)")
    parser.add_argument(
        "filename",
        nargs="?",
        default="../data/u_start_repaired_65536_variations_4_8_16_32_bus_grid_Ybus.parquet",
        help="Path to Parquet file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    path = Path(args.filename)
    if not path.exists():
        parser.error(f"File not found: {path}")

    # --------------------------- load ---------------------------
    df = pd.read_parquet(path)
    print(f"Original shape: {df.shape}")

    # ----------------- remove diverged Newton rows --------------
    mask_diverged = df.apply(row_has_zero_voltage, axis=1)
    if mask_diverged.any():
        df = df[~mask_diverged].reset_index(drop=True)
        print(f"Removed {mask_diverged.sum()} diverged rows → new shape: {df.shape}")

    # ensure U_base & S_base exist
    if {"U_base", "S_base"}.issubset(df.columns):
        u_base_col = df["U_base"].astype(float).to_numpy()
        s_base_col = df["S_base"].astype(float).to_numpy()
    else:
        raise KeyError("Parquet must contain 'U_base' and 'S_base' columns")

    # per‑row bases
    z_base_col = u_base_col ** 2 / s_base_col
    y_base_col = 1.0 / z_base_col

    # ------------------------------------------------------------------
    # streaming stats per column (physical & pu)
    # ------------------------------------------------------------------
    stats_phys: dict[str, dict[str, float]] = defaultdict(lambda: {
        "min": float("inf"), "max": -float("inf"), "sum": 0.0, "sum2": 0.0, "count": 0})
    stats_pu: dict[str, dict[str, float]] = defaultdict(lambda: {
        "min": float("inf"), "max": -float("inf"), "sum": 0.0, "sum2": 0.0, "count": 0})

    # global |V| & θ ranges
    v_min = float("inf"); v_max = -float("inf")
    v_pu_min = float("inf"); v_pu_max = -float("inf")
    th_min = float("inf"); th_max = -float("inf")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning rows"):
        u_base = float(u_base_col[idx])
        s_base = float(s_base_col[idx])
        z_base = float(z_base_col[idx])
        y_base = float(y_base_col[idx])

        # ---- per‑column processing -----------------------------------
        for col, cell in row.items():
            values = _tolist(cell)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            # physical stats
            s = stats_phys[col]
            s["min"] = min(s["min"], arr.min())
            s["max"] = max(s["max"], arr.max())
            s["sum"] += arr.sum(); s["sum2"] += (arr ** 2).sum(); s["count"] += arr.size

            # per‑unit conversion if base known
            divisor = infer_divisor(col, u_base, s_base, y_base)
            if divisor is not None and divisor != 0:
                arr_pu = arr / divisor
                sp = stats_pu[col]
                sp["min"] = min(sp["min"], arr_pu.min())
                sp["max"] = max(sp["max"], arr_pu.max())
                sp["sum"] += arr_pu.sum(); sp["sum2"] += (arr_pu ** 2).sum(); sp["count"] += arr_pu.size

        # ---- magnitude / angle --------------------------------------
        u_real = _tolist(row.get("u_newton_real"))
        u_imag = _tolist(row.get("u_newton_imag"))
        if u_real and u_imag:
            u_complex = np.asarray(u_real) + 1j * np.asarray(u_imag)
            mags = np.abs(u_complex); angs = np.angle(u_complex)
            v_min = min(v_min, mags.min()); v_max = max(v_max, mags.max())
            th_min = min(th_min, angs.min()); th_max = max(th_max, angs.max())
            if u_base != 0:
                mags_pu = mags / u_base
                v_pu_min = min(v_pu_min, mags_pu.min()); v_pu_max = max(v_pu_max, mags_pu.max())

    # ------------------------------------------------------------------
    # final report
    # ------------------------------------------------------------------
    def _print_stats(title: str, table: dict[str, dict[str, float]]):
        print(f"\n=== {title} ===")
        for col, s in table.items():
            n = s["count"]
            if n == 0:
                continue
            mean = s["sum"] / n; var = s["sum2"] / n - mean ** 2; std = sqrt(max(var, 0.0))
            print(f"{col:20s}  min = {s['min']:.6e},  max = {s['max']:.6e},  "
                  f"mean = {mean:.6e},  std = {std:.6e}")

    _print_stats("Physical units", stats_phys)
    _print_stats("Per‑unit (p.u.)", stats_pu)

    # voltage / angle summary
    print("\n=== |V| & θ ranges ===")
    print(f"|V| physical : [{v_min:.6e}, {v_max:.6e}]")
    print(f"|V| per‑unit  : [{v_pu_min:.6e}, {v_pu_max:.6e}]")
    print(f"θ  (radians) : [{th_min:.6e}, {th_max:.6e}]")


if __name__ == "__main__":
    main()
