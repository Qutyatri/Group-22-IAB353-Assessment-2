# Professional data cleaning using pandas only
# Extracts 'year' from 'education', normalizes text, replaces 'unknown' with 'N/A',
# expands weekdays to full names, and rounds specified columns.

import pandas as pd

INPUT_PATH  = "bank-additional-full.csv"
OUTPUT_PATH = "bank_cleaned_final.csv"
SEP = ";"

# Load dataset
df = pd.read_csv(INPUT_PATH, sep=SEP)

# Replace 'unknown' with 'N/A'
df = df.replace('unknown', 'N/A')

# Create 'year' column from education (e.g., 'basic 4y' → '4', else 'N/A')
edu_series = df['education'].astype(str)
df['year'] = edu_series.str.extract(r'(\d+)(?=y)', expand=False).fillna('N/A')

# Clean 'education' text (remove numeric part, keep only base level)
df['education'] = (
    edu_series
    .str.replace(r'[\.\s]*\d+\s*y', '', regex=True)
    .str.replace(r'\.', '', regex=True)
    .str.strip()
    .str.lower()
)
df.loc[df['education'].isin(['', 'nan']), 'education'] = 'N/A'

# Move 'year' column right after 'education'
cols = df.columns.tolist()
if 'education' in cols and 'year' in cols:
    edu_idx = cols.index('education')
    cols.insert(edu_idx + 1, cols.pop(cols.index('year')))
    df = df[cols]

# Expand abbreviated weekdays (mon → monday)
dow_map = {
    'mon': 'monday', 'tue': 'tuesday', 'wed': 'wednesday',
    'thu': 'thursday', 'fri': 'friday', 'sat': 'saturday', 'sun': 'sunday'
}
if 'day_of_week' in df.columns:
    s = df['day_of_week'].astype(str).str.strip().str.lower()
    df['day_of_week'] = s.map(dow_map).fillna(s)

# Round specific numeric columns to 2 decimals
for col in ['cons.price.idx', 'euribor3m']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Cleaned dataset saved → {OUTPUT_PATH}")
print("Shape:", df.shape)
