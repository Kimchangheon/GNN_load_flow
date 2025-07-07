
import pandas as pd

filename = r'../data/u_start_repaired_1_variations_4_8_16_32_bus_grid_Ybus.parquet'
# read the whole file
df = pd.read_parquet(filename)
print(df.columns)       # e.g. (number_of_rows, number_of_columns)

# inspect it
print(df.shape)       # e.g. (number_of_rows, number_of_columns)
print(df.dtypes)      # column data types
print(df.head())      # first five rows