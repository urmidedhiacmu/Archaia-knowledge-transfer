import pandas as pd

print("Loading merged dataset...")

df = pd.read_parquet("data/artifacts_with_spacetime.parquet")

print("Rows:", df.shape)

# -------------------------
# QUALITY SCORING FUNCTION
# -------------------------

def score_row(row):
    score = 0

    # Direct reference is best
    if row.get("reference_type") == "direct":
        score += 5

    # Has coordinates
    if pd.notna(row.get("latitude")):
        score += 3

    # Has time
    if pd.notna(row.get("start")):
        score += 3

    # Shallower depth better
    depth = row.get("geo_depth")
    if pd.notna(depth):
        score += max(0, 2 - depth)

    return score


print("Scoring rows...")
df["quality_score"] = df.apply(score_row, axis=1)

# -------------------------
# FIND BEST PER ARTIFACT
# -------------------------

print("Marking best rows...")

df["is_best"] = False

best_idx = (
    df.sort_values("quality_score", ascending=False)
      .groupby("uuid_hex")
      .head(1)
      .index
)

df.loc[best_idx, "is_best"] = True

# -------------------------
# SAVE
# -------------------------

df.to_parquet(
    "data/artifacts_with_spacetime_ranked.parquet",
    index=False
)

df.to_csv(
    "data/artifacts_with_spacetime_ranked.csv",
    index=False
)

print("Saved ranked dataset ✅")

print("Artifacts with at least one location:",
      df[df["latitude"].notna()]["uuid_hex"].nunique())

print("Artifacts with best location:",
      df[df["is_best"]]["uuid_hex"].nunique())
