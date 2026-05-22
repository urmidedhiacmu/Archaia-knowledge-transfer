#!/usr/bin/env python3
import re
import html
import pandas as pd

INFILE = "test_recovered_assertions_subset.csv"
OUT_LONG = "test_recovered_assertions_subset_cleaned.csv"
OUT_WIDE = "test_recovered_assertions_subset_wide.csv"

# Broad predicate filter for fields we actually care about for semantic search / natural language recovery
KEEP_RE = re.compile(
    r"(note|description|material|artifact name|object type|object type \(notes\)|"
    r"period|chronotype|fabric|ware|color|decorative|surface|worked|vessel form|"
    r"function|class|condition|size|munsell|location|locus|context|trench)",
    re.IGNORECASE
)

def strip_html(text):
    if pd.isna(text):
        return text
    s = str(text)
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)   # remove tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = pd.read_csv(INFILE)

# Keep only meaningful descriptive predicates
df = df[df["predicate"].astype(str).str.contains(KEEP_RE, na=False)].copy()

# Clean values
df["value_clean"] = df["value"].apply(strip_html)

# Drop exact duplicates
df = df.drop_duplicates(subset=["uuid_hex", "predicate", "value_clean"]).copy()

print("Filtered rows:", len(df))
print("\nSample:")
print(df.head(40).to_string(index=False))

df.to_csv(OUT_LONG, index=False)
print(f"\nSaved long cleaned preview: {OUT_LONG}")

# Make a wide preview per artifact
grouped = (
    df.groupby(["uuid_hex", "predicate"])["value_clean"]
      .apply(lambda s: " || ".join(dict.fromkeys([str(x) for x in s if pd.notna(x)])))
      .reset_index()
)

wide = grouped.pivot(index="uuid_hex", columns="predicate", values="value_clean").reset_index()
wide.to_csv(OUT_WIDE, index=False)
print(f"Saved wide preview: {OUT_WIDE}")

print("\nWide preview columns:")
print(list(wide.columns))
print("\nWide preview sample:")
print(wide.head(10).to_string(index=False))
