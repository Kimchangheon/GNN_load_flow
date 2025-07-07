import pandas as pd

filename = r'../data/u_start_repaired_65536_variations_4_8_16_32_bus_grid_Ybus.parquet'

# Read only the first 800 rows
df = pd.read_parquet(filename)
df_first800 = df.head(800)

# Save to a new Parquet file
output_filename = r'../data/u_start_repaired_800_variations_4_8_16_32_bus_grid_Ybus.parquet'
df_first800.to_parquet(output_filename, index=False)

print(f"Saved first 800 rows to {output_filename}")