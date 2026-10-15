import pandas as pd

df = pd.read_csv("data/census.csv")
cat_cols = df.select_dtypes(include="object").columns

print("=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column types ===")
print(df.dtypes)

print("\n=== Missing values (NaN) per column ===")
print(df.isna().sum())

print("\n=== '?' placeholders per column ===")
print((df[cat_cols] == "?").sum())

print("\n=== Stray spaces in column names or values? ===")
print("column names:", [c for c in df.columns if c != c.strip()])
print("values:", [c for c in cat_cols if df[c].str.startswith(" ").any()])

print("\n=== Duplicate rows ===")
print(df.duplicated().sum())

print("\n=== Label balance (salary) ===")
print(df["salary"].value_counts(normalize=True).round(3))

print("\n=== Unique values per categorical column ===")
print(df[cat_cols].nunique())

print("\n=== Numeric summary ===")
print(df.describe().round(1).T)
