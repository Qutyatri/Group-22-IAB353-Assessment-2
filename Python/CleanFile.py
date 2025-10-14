# Professional data cleaning (pandas-only)
# Keeps 'duration', keeps y as "yes"/"no", adds 'year' beside 'education'
# Keeps age as actual value (not normalized)
# 2-decimal numeric output, uppercase "N/A"

import pandas as pd

INPUT_PATH  = "bank-additional-full.csv"
OUTPUT_PATH = "bank_cleaned_final1.csv"
SEP = ";"

KEEP_COLS = [
    "age","job","marital","education","default","housing","loan",
    "contact","month","day_of_week","duration","campaign","pdays","previous","poutcome",
    "emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"
]

NUMERIC_COLS = [
    "duration","campaign","pdays","previous",
    "emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"
]  # age excluded from scaling

df = pd.read_csv(INPUT_PATH, sep=SEP)
df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

df = df.replace(r'^\s*$', pd.NA, regex=True).replace('unknown', pd.NA)

cat_cols = [c for c in df.columns if c not in NUMERIC_COLS + ["y","age"]]
for c in cat_cols:
    s = df[c].astype(str).str.strip().str.lower()
    s = s.mask(s.isin(["nan","none","nat"]), pd.NA)
    df[c] = s

# Create 'year' from education and simplify education text
df["year"] = df["education"].astype(str).str.extract(r'(\d+)')
df["year"] = df["year"].fillna("N/A")
df["education"] = df["education"].astype(str).str.replace(r'[\.\d+y]', '', regex=True).str.strip()
df.loc[df["education"].isin(["", "nan", "none", "nat"]), "education"] = "N/A"

# Place 'year' beside 'education'
cols = df.columns.tolist()
if "education" in cols and "year" in cols:
    edu_i = cols.index("education")
    cols.insert(edu_i + 1, cols.pop(cols.index("year")))
    df = df[cols]

# Convert numerics
for c in NUMERIC_COLS + ["age"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "pdays" in df.columns:
    df["pdays"] = df["pdays"].mask(df["pdays"] == 999, pd.NA)

# Impute missing
for c in df.columns:
    if c in NUMERIC_COLS + ["age"]:
        df[c] = df[c].fillna(df[c].median())
    elif c != "y":
        mode_val = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "N/A")

# Outlier capping
for c in NUMERIC_COLS + ["age"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[c] = df[c].clip(lb, ub)

# Scale all numeric columns EXCEPT age
for c in NUMERIC_COLS:
    mn, mx = df[c].min(), df[c].max()
    df[c] = 0.0 if pd.isna(mn) or pd.isna(mx) or mx == mn else (df[c] - mn) / (mx - mn)

# Round numeric values to two decimals
float_cols = df.select_dtypes(include=["float64","float32"]).columns
df[float_cols] = df[float_cols].round(2)

# Replace NA with uppercase "N/A"
df = df.fillna("N/A")
obj_cols = df.select_dtypes(include=["object"]).columns
for c in obj_cols:
    df[c] = df[c].replace(to_replace=r'(?i)^\s*n/?a\s*$', value="N/A", regex=True)

# Keep y as yes/no or N/A
if "y" in df.columns:
    df["y"] = df["y"].astype(str).str.strip().str.lower()
    df["y"] = df["y"].where(df["y"].isin(["yes","no"]), other="N/A")

df.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")
print(f"Saved: {OUTPUT_PATH} | Shape: {df.shape}")
