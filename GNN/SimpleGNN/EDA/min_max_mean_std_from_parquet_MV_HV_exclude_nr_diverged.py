#!/usr/bin/env python3
"""
Compute per-column min, max, mean, std from a Parquet file **after
removing diverged Newton–Raphson cases** (rows where both
`u_newton_real` and `u_newton_imag` are all-zero).

Usage
-----
    python stats_no_diverged.py path/to/file.parquet
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

_NUMBER_RE = re.compile(
    r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)  # quick numeric-string test


def _tolist(cell) -> list[float]:
    """Return a *flat list of floats* contained in *cell* (see original docstring)."""
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
    """True if *both* u_newton columns in this row are lists / arrays of *all* zeros."""
    real_vals = _tolist(row["u_newton_real"])
    imag_vals = _tolist(row["u_newton_imag"])
    return real_vals and imag_vals and all(v == 0 for v in real_vals + imag_vals)


###############################################################################
# main logic
###############################################################################

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="Column-wise statistics (diverged rows removed)")
    parser.add_argument(
        "filename",
        nargs="?",
        default="../data/u_start_repaired_65536_variations_4_8_16_32_bus_grid_Ybus.parquet",
        help="Path to the Parquet file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    filename = Path(args.filename)
    if not filename.exists():
        parser.error(f"File not found: {filename}")

    # ---------------------------- load dataframe -----------------------------
    df = pd.read_parquet(filename)
    print(f"Original shape: {df.shape}")

    # ------------------------ drop diverged Newton rows ---------------------
    mask_diverged = df.apply(row_has_zero_voltage, axis=1)
    n_removed = mask_diverged.sum()
    if n_removed:
        df = df[~mask_diverged].reset_index(drop=True)
        print(f"Removed {n_removed} diverged rows → new shape: {df.shape}")
    else:
        print("No diverged rows found")

    # ------------------------- streaming statistics -------------------------
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"min": float("inf"), "max": -float("inf"), "sum": 0.0, "sum2": 0.0, "count": 0}
    )

    for col in tqdm(df.columns, desc="Scanning columns"):
        col_data = df[col]

        # fast path: purely numeric scalars
        if pd.api.types.is_numeric_dtype(col_data) and not col_data.apply(
            lambda x: isinstance(x, (list, tuple, np.ndarray, str))
        ).any():
            arr = col_data.to_numpy(dtype=np.float64, copy=False)
            arr = arr[np.isfinite(arr)]
        else:
            flat: list[float] = []
            for cell in col_data:
                flat.extend(_tolist(cell))
            if not flat:
                continue
            arr = np.asarray(flat, dtype=np.float64)
            arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            continue

        s = stats[col]
        s["min"] = min(s["min"], arr.min())
        s["max"] = max(s["max"], arr.max())
        s["sum"] += arr.sum()
        s["sum2"] += (arr ** 2).sum()
        s["count"] += arr.size

    # ------------------------------ final report -----------------------------
    print("\n=== Column-wise statistics (after cleaning) ===")
    for col, s in stats.items():
        n = s["count"]
        if n == 0:
            print(f"{col:20s}  (no numeric data)")
            continue
        mean = s["sum"] / n
        var = s["sum2"] / n - mean ** 2
        std = sqrt(max(var, 0.0))
        print(
            f"{col:20s}  min = {s['min']:.6e},  max = {s['max']:.6e},  "
            f"mean = {mean:.6e},  std = {std:.6e}"
        )


if __name__ == "__main__":
    main()