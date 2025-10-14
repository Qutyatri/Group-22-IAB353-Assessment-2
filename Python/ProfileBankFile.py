import pandas as pd

# Load the dataset
df = pd.read_csv("bank-additional-full.csv", sep=';')

# Basic structure
print("Shape:", df.shape)
print("\n--- Data Info ---")
print(df.info())
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Summary of numeric columns ---")
print(df.describe().T)

print("\n--- Target variable (y) distribution ---")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True)*100)

# True missing values
print("\n--- True missing values ---")
print(df.isna().sum())

# Encoded 'unknown' values
print("\n--- 'unknown' value counts ---")
for col in df.columns:
    if df[col].dtype == 'object':
        unknown_count = (df[col] == 'unknown').sum()
        if unknown_count > 0:
            print(f"{col}: {unknown_count}")

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

corr = df[numeric_cols].corr()
print("\n--- Correlation matrix ---")
print(corr.round(2))

