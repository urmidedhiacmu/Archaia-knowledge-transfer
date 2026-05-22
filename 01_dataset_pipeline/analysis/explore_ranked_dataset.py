import pandas as pd
import numpy as np

PATH = "data/artifacts_best_spacetime.csv"
# PATH = "data/artifacts_with_spacetime_ranked.csv"

print("\nLoading dataset...")
df = pd.read_csv(PATH, low_memory=False)

print("\n================ BASIC INFO ================")
print("Shape:", df.shape)
print("Columns:", len(df.columns))
print(df.columns.tolist())

print("\n================ UNIQUE COUNTS ================")
print("Unique artifacts:", df["uuid_hex"].nunique())
print("Total rows:", len(df))

if "is_best" in df.columns:
    print("Best rows:", df["is_best"].sum())

print("\n================ LOCATION COVERAGE ================")

has_loc = df["latitude"].notna()
has_time = df["start"].notna()

print("Rows with location:", has_loc.sum())
print("Rows with time:", has_time.sum())

print("Artifacts with location:",
      df[has_loc]["uuid_hex"].nunique())

print("Artifacts with time:",
      df[has_time]["uuid_hex"].nunique())

print("\n================ CANDIDATES PER ARTIFACT ================")

counts = df.groupby("uuid_hex").size()

print("Mean candidates:", counts.mean())
print("Median candidates:", counts.median())
print("Max candidates:", counts.max())

print("\nTop 10 artifacts with most candidates:")
print(counts.sort_values(ascending=False).head(10))

print("\n================ MISSING VALUES (%) ================")

missing = df.isna().mean().sort_values(ascending=False) * 100
print(missing.head(20))

print("\n================ GEO DEPTH DISTRIBUTION ================")

if "geo_depth" in df.columns:
    print(df["geo_depth"].value_counts(dropna=False))

print("\n================ CHRONO DEPTH DISTRIBUTION ================")

if "chrono_depth" in df.columns:
    print(df["chrono_depth"].value_counts(dropna=False))

print("\n================ QUALITY SCORE ================")

if "quality_score" in df.columns:
    print(df["quality_score"].describe())

print("\n================ BEST ROW QUALITY ================")

if "is_best" in df.columns:
    best = df[df["is_best"] == True]

    print("Best rows:", len(best))

    if "quality_score" in best.columns:
        print(best["quality_score"].describe())

print("\n================ LAT/LON RANGE ================")

if "latitude" in df.columns:
    print("Latitude:", df["latitude"].min(), "→", df["latitude"].max())

if "longitude" in df.columns:
    print("Longitude:", df["longitude"].min(), "→", df["longitude"].max())

print("\n================ TIME RANGE ================")

if "start" in df.columns:
    print("Start:", df["start"].min(), "→", df["start"].max())

if "stop" in df.columns:
    print("Stop:", df["stop"].min(), "→", df["stop"].max())

print("\n================ DUPLICATE CHECK ================")

dups = df.duplicated(subset=["uuid_hex", "latitude", "longitude", "start", "stop"]).sum()
print("Exact duplicates:", dups)

print("\n================ SAMPLE ROWS ================")
print(df.head())

print("\nDone.")
