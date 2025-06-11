import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Parquet file
df = pd.read_parquet("../data/212100_variations_4_8_16_32_bus_grid.parquet")

# 1) Histogram of each bus_number
plt.figure(figsize=(6,4))
df['bus_number'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Number of Buses in Grid')
plt.ylabel('Count of Samples')
plt.title('Histogram of Grid Sizes (bus_number)')
plt.tight_layout()
plt.show()

# 2) Unique bus_type lists per grid size
print("Unique bus_type configurations by bus_number:")
for n, group in df.groupby('bus_number'):
    # each row's bus_typ is a list; flatten and get unique patterns
    patterns = group['bus_typ'].apply(tuple).unique()
    print(f" • {n} buses: {list(patterns)}")

# 3) Unique Lines_connected patterns per grid size
print("\nUnique connection‐vector patterns by bus_number:")
for n, group in df.groupby('bus_number'):
    conns = group['Lines_connected'].apply(tuple).unique()
    print(f" • {n} buses: {len(conns)} distinct patterns")

# If you want to *see* one example of each:
#    for pat in conns: print("   ", pat)

# 4) Unique U_base values across all samples
print("\nUnique U_base values in dataset:")
print(df['U_base'].unique())


# 4) see 4-bus cases
# 2. Filter to only the 4-bus cases
df4 = df[df["bus_number"] == 4]
# 3. Select and print the voltage columns
cols = ["bus_typ","Lines_connected","u_newton_real", "u_newton_imag", "u_start_real", "u_start_imag"]
print("Values for bus_number == 4:\n")
print(df4[cols].to_string(index=False))


# 4) see non-converged cases
# Find rows where every entry in u_newton_real is exactly 0.0
def is_all_zero(lst):
    return all(float(v) == 0.0 for v in lst)

mask = df['u_newton_real'].apply(is_all_zero)

# Extract the indices and bus_number for those samples
zero_rows = df[mask]

print("Rows where u_newton_real is all zeros:")
print("total :", len(zero_rows)) # 59048 out of 212100
# for idx, row in zero_rows.iterrows():
#     print(f"  DataFrame row index: {idx}, bus_number: {row['bus_number']}")