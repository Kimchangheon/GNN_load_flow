#!/usr/bin/env python3
"""
Compute per‑column min, max, mean and standard deviation from a Parquet file
where cells can contain scalars, sequences, NumPy arrays or even *stringified*
lists.  The script is resilient against malformed string cells: anything that
cannot be interpreted as a numeric value (or a list / tuple of numerics) is
ignored instead of crashing.

Usage
-----
    python min_max_mean_std_from_parquet_adapted.py path/to/file.parquet

If no filename is given it falls back to the hard‑coded relative path that the
original script used.
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

# pre‑compiled regexp to recognise simple floating‑point numbers in string form
_NUMBER_RE = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$")


def _tolist(cell) -> list[float]:
    """Return a *list of floats* extracted from *cell*.

    The function accepts several input varieties and converts them to a flat
    list so the calling code can process all columns uniformly.

    Supported input types
    ---------------------
    * *None* / *NaN* → empty list
    * `int`, `float`, NumPy scalar → one‑element list
    * `list`, `tuple`, `np.ndarray` → converted with `list()` and flattened one level
    * *string*        → interpreted **iff** it represents a list/tuple *or* a single
                        number; anything else is treated as “no numeric data”
    """
    # ------------------------------------------------------------------ nulls
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    # ---------------------------------------------------------- already list‑like
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)

    # ----------------------------------------------------------- plain numerics
    if isinstance(cell, (int, float, np.number)):
        return [cell]

    # ---------------------------------------------------------------- strings
    if isinstance(cell, str):
        s = cell.strip()

        # 1) string *looks* like a Python/JSON list or tuple → try literal_eval
        if s and s[0] in "[(":
            try:
                return list(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                # fall through to the rest of the checks
                pass

        # 2) string looks like a single numeric value
        if _NUMBER_RE.match(s):
            try:
                return [float(s)]
            except ValueError:
                pass  # fall through

        # 3) everything else → treat as no data
        return []

    # --------------------------------------------------------------- fallback
    # Anything that reaches here is an unexpected type
    raise TypeError(f"Unsupported cell type encountered: {type(cell)}")


###############################################################################
# main logic
###############################################################################

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="Column‑wise statistics from a Parquet file")
    parser.add_argument("filename", nargs="?", default="../data/Changhun_multi_3_7000.parquet",
                        help="Path to the Parquet file (default: %(default)s)")
    args = parser.parse_args(argv)

    filename = Path(args.filename)
    if not filename.exists():
        parser.error(f"File not found: {filename}")

    # ---------------------------- load dataframe -----------------------------
    df = pd.read_parquet(filename)
    print(f"DataFrame shape: {df.shape}")

    # ------------------------- streaming statistics -------------------------
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "min": float("inf"),
        "max": -float("inf"),
        "sum": 0.0,
        "sum2": 0.0,
        "count": 0,
    })

    for col in tqdm(df.columns, desc="Scanning columns"):
        col_data = df[col]

        # Fast path: purely numeric column with scalar entries only
        if pd.api.types.is_numeric_dtype(col_data) and not col_data.apply(
            lambda x: isinstance(x, (list, tuple, np.ndarray, str))
        ).any():
            arr = col_data.to_numpy(dtype=np.float64, copy=False)
            arr = arr[np.isfinite(arr)]  # drop NaN/inf
        else:
            # Need to flatten the mixed contents
            flat: list[float] = []
            for cell in col_data:
                flat.extend(_tolist(cell))
            if not flat:
                continue  # column has no numeric data at all
            arr = np.asarray(flat, dtype=np.float64)
            arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            continue  # nothing numeric in this column

        s = stats[col]
        s["min"] = min(s["min"], arr.min())
        s["max"] = max(s["max"], arr.max())
        s["sum"] += arr.sum()
        s["sum2"] += (arr ** 2).sum()
        s["count"] += arr.size

    # ------------------------------ final report -----------------------------
    print("\n=== Column‑wise statistics ===")
    for col, s in stats.items():
        n = s["count"]
        if n == 0:
            print(f"{col:20s}  (no numeric data)")
            continue
        mean = s["sum"] / n
        var = s["sum2"] / n - mean ** 2
        std = sqrt(max(var, 0.0))  # guard tiny negatives due to FP error
        print(
            f"{col:20s}  min = {s['min']:.6e},  max = {s['max']:.6e},  "
            f"mean = {mean:.6e},  std = {std:.6e}"
        )


if __name__ == "__main__":
    main()
