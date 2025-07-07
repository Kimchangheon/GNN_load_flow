import pandas as pd
import numpy as np
import ast
from math import sqrt
from collections import defaultdict
from tqdm import tqdm   # optional progress bar

# ------------------------ helpers ---------------------------------
def _tolist(cell):
    """
    Return a Python list of (float) numbers from any cell that may be:
       – a Python list / tuple / np.ndarray
       – a scalar (int/float)                → wrapped in one-element list
       – a string representation of a list   → ast.literal_eval
       – None / NaN                          → empty list
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    if isinstance(cell, (int, float, np.number)):
        return [cell]
    if isinstance(cell, str):
        return list(ast.literal_eval(cell))
    raise TypeError(f"Unsupported cell type: {type(cell)}")  # unexpected

# ----------------------- load dataframe ---------------------------
filename = "../data/Changhun_multi_3_7000.parquet"
df = pd.read_parquet(filename)
print(f"DataFrame shape: {df.shape}")

# ----------------------- running stats ----------------------------
stats = defaultdict(lambda: {
    "min":  float("inf"),
    "max": -float("inf"),
    "sum":  0.0,
    "sum2": 0.0,
    "count": 0
})

for col in tqdm(df.columns, desc="Scanning columns"):
    col_data = df[col]

    # Fast path for purely numeric columns (pandas dtype is number *and* each cell scalar)
    if pd.api.types.is_numeric_dtype(col_data) and not col_data.apply(lambda x: isinstance(x, (list, tuple, np.ndarray, str))).any():
        arr = col_data.to_numpy(dtype=np.float64)
        # drop NaNs / infs
        arr = arr[np.isfinite(arr)]
    else:
        # need to flatten list-like cells
        flat = []
        for cell in col_data:
            flat.extend(_tolist(cell))
        if not flat:               # avoid zero-length arrays
            continue
        arr = np.asarray(flat, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

    # update streaming aggregates
    s = stats[col]
    if arr.size == 0:         # skip columns that ended up empty
        continue
    s["min"]   = min(s["min"], arr.min())
    s["max"]   = max(s["max"], arr.max())
    s["sum"]  += arr.sum()
    s["sum2"] += (arr**2).sum()
    s["count"] += arr.size

# ----------------------- final report -----------------------------
print("\n=== Column-wise statistics ===")
for col, s in stats.items():
    N = s["count"]
    if N == 0:
        print(f"{col:15s}  (no numeric data)")
        continue
    mean = s["sum"]  / N
    var  = s["sum2"] / N - mean**2
    std  = sqrt(max(var, 0.0))      # guard tiny negatives
    print(f"{col:15s}  "
          f"min = {s['min']: .6e},  "
          f"max = {s['max']: .6e},  "
          f"mean = {mean: .6e},  "
          f"std = {std: .6e}")