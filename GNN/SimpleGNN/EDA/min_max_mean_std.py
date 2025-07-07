import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from GNN.SimpleGNN.Dataset import ChanghunDataset   # ← adjust import path if needed
from math import sqrt
from tqdm import tqdm   # progress bar (optional)
# ------------------------------------------------------------
# 1.  Load the dataset
# ------------------------------------------------------------
PARQUET = "../data/212100_variations_4_8_16_32_bus_grid.parquet"   # <-- your file
ds = ChanghunDataset(PARQUET)

# ------------------------------------------------------------------
# 2.  Prepare running accumulators
#     (one set per key in every sample dict, except the scalar 'N')
# ------------------------------------------------------------------
stats = defaultdict(lambda: {
    "min":  float("inf"),
    "max": -float("inf"),
    "sum":  0.0,
    "sum2": 0.0,
    "count": 0
})

# ------------------------------------------------------------------
# 3.  Sweep the dataset once, updating the streaming statistics
# ------------------------------------------------------------------
for sample in tqdm(ds, desc="Scanning dataset"):
    for k, v in sample.items():
        if k == "N":          # skip the scalar size indicator
            continue

        # convert to float32 tensor (handles int, float, complex split magn/angle)
        t = v.float()

        s = stats[k]
        s["min"]   = min(s["min"], t.min().item())
        s["max"]   = max(s["max"], t.max().item())
        s["sum"]  += t.sum().item()
        s["sum2"] += (t**2).sum().item()
        s["count"] += t.numel()

# ------------------------------------------------------------------
# 4.  Finalise: compute mean and std, then print nicely
# ------------------------------------------------------------------
print("\n=== Dataset statistics ===")
for k, s in stats.items():
    mean = s["sum"]  / s["count"]
    var  = s["sum2"] / s["count"] - mean**2
    std  = sqrt(max(var, 0.0))            # guard against tiny negatives

    print(f"{k:10s}  "
          f"min = {s['min']: .6e},  "
          f"max = {s['max']: .6e},  "
          f"mean = {mean: .6e},  "
          f"std = {std: .6e}")