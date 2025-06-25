
import pandas as pd

filename = r'labels.csv'
# read the whole file
df = pd.read_csv(filename)

# First row
row1 = df.iloc[0]
print("\nRow 1:")
for column in df.columns:
    value = row1[column]
    if isinstance(value, str):
        # Split the string by comma and convert to list
        value_list = value.strip('[]').split(',')
        # Remove any empty strings that might result from splitting
        value_list = [x.strip() for x in value_list if x.strip()]
        print(f"{column}, Length: {len(value_list)}")
    else:
        print(f"{column}, Length: 1 (non-string value)")

# Second row
row2 = df.iloc[1]
print("\nRow 2:")
for column in df.columns:
    value = row2[column]
    if isinstance(value, str):
        value_list = value.strip('[]').split(',')
        value_list = [x.strip() for x in value_list if x.strip()]
        print(f"Column: {column}, Length: {len(value_list)}")
    else:
        print(f"Column: {column}, Length: 1 (non-string value)")

# Third row
row3 = df.iloc[2]
print("\nRow 3:")
for column in df.columns:
    value = row3[column]
    if isinstance(value, str):
        value_list = value.strip('[]').split(',')
        value_list = [x.strip() for x in value_list if x.strip()]
        print(f"Column: {column}, Length: {len(value_list)}")
    else:
        print(f"Column: {column}, Length: 1 (non-string value)")


print(df.columns)       # e.g. (number_of_rows, number_of_columns)

# inspect it
print(df.shape)       # e.g. (number_of_rows, number_of_columns)
print(df.dtypes)      # column data types
print(df.head())      # first five rows