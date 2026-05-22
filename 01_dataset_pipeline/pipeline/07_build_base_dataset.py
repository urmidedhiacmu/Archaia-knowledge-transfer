import os
import glob
import pandas as pd

USER = os.getenv("USER")

BASE = f"/data/user_data/{USER}/archaia"
IMG_DIRS = [
    f"{BASE}/artifact_images_1024",
    f"{BASE}/artifact_images_1024_full",
]

MAPPING_FILE = f"{BASE}/img2dataset_download.tsv"
SPACETIME_FILE = "/home/udedhia/archaia_project/data/artifacts_with_spacetime_ranked.csv"

OUT_DIR = f"{BASE}/final"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DATASET_PARQUET = f"{OUT_DIR}/archaia_final_dataset.parquet"
OUT_DATASET_CSV = f"{OUT_DIR}/archaia_final_dataset.csv"
OUT_MAPPING = f"{OUT_DIR}/final_downloaded_image_mapping.csv"
OUT_STATS = f"{OUT_DIR}/dataset_stats.txt"


# --------------------------------------------------
# STEP 1 — Scan images
# --------------------------------------------------

print("\nSTEP 1 — Scanning images")

all_indices = set()

for d in IMG_DIRS:
    files = glob.glob(f"{d}/*/*.jpg")
    print(f"{d}: {len(files):,}")

    for f in files:
        name = os.path.basename(f).split(".")[0]
        try:
            idx = int(name)
            all_indices.add(idx)
        except:
            pass

print(f"\nTotal unique image indices: {len(all_indices)}")


# --------------------------------------------------
# STEP 2 — Load mapping
# --------------------------------------------------

print("\nSTEP 2 — Loading mapping")

map_df = pd.read_csv(
    MAPPING_FILE,
    sep="\t",
    header=None,
    names=["url", "key"],
    dtype=str
)

print(f"Mapping rows: {len(map_df):,}")

map_df["img_index"] = map_df["key"].str.extract(r"_(\d+)$")[0]
map_df["img_index"] = pd.to_numeric(map_df["img_index"], errors="coerce")


# --------------------------------------------------
# STEP 3 — Match downloaded images
# --------------------------------------------------

print("\nSTEP 3 — Matching indices")

downloaded_df = map_df[map_df["img_index"].isin(all_indices)].copy()

print(f"Downloaded rows: {len(downloaded_df):,}")

downloaded_df["artifact_id"] = downloaded_df["key"].str.replace(
    r"_(\d+)$", "", regex=True
)

artifact_counts = (
    downloaded_df.groupby("artifact_id")
    .size()
    .reset_index(name="image_count")
)

print(f"Artifacts with ≥1 image: {len(artifact_counts):,}")

downloaded_df.to_csv(OUT_MAPPING, index=False)


# --------------------------------------------------
# STEP 4 — Load spacetime dataset
# --------------------------------------------------

print("\nSTEP 4 — Loading spacetime dataset")

df = pd.read_csv(SPACETIME_FILE, low_memory=False)

print(f"Rows before filtering: {len(df):,}")

df = df[df["is_best"] == True].copy()

print(f"Rows after is_best filter: {len(df):,}")

# 🔥 IMPORTANT FIX
df["artifact_id"] = "artifact_" + df["uuid_hex"].astype(str)


# --------------------------------------------------
# STEP 5 — Merge with image counts
# --------------------------------------------------

print("\nSTEP 5 — Merging images")

final_df = df.merge(
    artifact_counts,
    on="artifact_id",
    how="inner"
)

print(f"Final artifacts: {len(final_df):,}")


# --------------------------------------------------
# STEP 6 — Save outputs
# --------------------------------------------------

print("\nSTEP 6 — Saving")

final_df.to_parquet(OUT_DATASET_PARQUET, index=False)
final_df.to_csv(OUT_DATASET_CSV, index=False)

with open(OUT_STATS, "w") as f:
    f.write("FINAL DATASET STATS\n")
    f.write("====================\n")
    f.write(f"images_total: {len(downloaded_df):,}\n")
    f.write(f"artifacts_with_images: {len(final_df):,}\n")

print("\n==============================")
print("FINAL DATASET COMPLETE")
print("==============================")
print(f"Images used: {len(downloaded_df):,}")
print(f"Artifacts final: {len(final_df):,}")
print(f"Saved to: {OUT_DIR}")