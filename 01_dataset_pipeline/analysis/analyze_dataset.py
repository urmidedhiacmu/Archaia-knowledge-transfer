#!/usr/bin/env python3
import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt


# ======================
# PATHS (ADJUST IF NEEDED)
# ======================
DATASET_PATH = "/home/udedhia/archaia_project/data/artifacts_with_spacetime_ranked.parquet"

# IMPORTANT: mapping is the one produced by your pipeline:
# columns (no header): artifact_image_id, artifact_uuid(hex), image_url, hash_bytes
MAPPING_PATH = "/data/user_data/udedhia/archaia/image_to_artifact_mapping.csv"

# IMPORTANT: this must match your img2dataset output dir (the one containing 00000/, 00001/, ... and *_stats.json)
IMAGE_ROOT = "/data/user_data/udedhia/archaia/artifact_images_1024_full"

OUTPUT_DIR = "ppt_outputs_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================
# UTIL: read URL from an img2dataset json
# ======================
def extract_url_from_img2dataset_json(obj: dict) -> str | None:
    """
    img2dataset json schemas vary slightly by version/output_format.
    We try common keys.
    """
    if not isinstance(obj, dict):
        return None

    # Common candidates seen across versions
    candidates = [
        "url",
        "original_url",
        "input_url",
        "requested_url",
        "source_url",
        "img_url",
        "image_url",
    ]
    for k in candidates:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Sometimes nested
    # e.g. {"meta": {"url": "..."}}
    meta = obj.get("meta")
    if isinstance(meta, dict):
        for k in candidates:
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


# ======================
# STEP 0: LOAD DATA
# ======================
print("Loading dataset parquet...")
df = pd.read_parquet(DATASET_PATH)

# You MUST join using uuid_hex (string) because uuid is bytes
if "uuid_hex" not in df.columns:
    raise RuntimeError("Dataset is missing uuid_hex column — cannot join to mapping safely.")

print("Loading image_to_artifact mapping...")
mapping = pd.read_csv(
    MAPPING_PATH,
    header=None,
    names=["artifact_image_id", "artifact_uuid", "image_url", "hash_bytes"],
)

# Basic sanity
mapping["artifact_uuid"] = mapping["artifact_uuid"].astype(str).str.strip()
mapping["image_url"] = mapping["image_url"].astype(str).str.strip()

# Some rows can be malformed; drop empties
mapping = mapping[(mapping["artifact_uuid"] != "") & (mapping["image_url"] != "")]
mapping = mapping.drop_duplicates(subset=["artifact_uuid", "image_url"])

print(f"Dataset rows: {len(df):,}")
print(f"Mapping rows: {len(mapping):,}")
print(f"Unique artifacts in mapping: {mapping['artifact_uuid'].nunique():,}")


# ======================
# STEP 1: BUILD URL -> LOCAL JPG INDEX (THE IMPORTANT PART)
# ======================
def build_url_to_local_jpg_index(image_root: str) -> dict[str, str]:
    """
    Walk IMAGE_ROOT and read each per-image json (NOT *_stats.json),
    extracting the URL and mapping it to the corresponding local .jpg path.
    This is the only non-guessy way.
    """
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"IMAGE_ROOT does not exist: {image_root}")

    # img2dataset creates shard dirs like 00000, 00001, ... and also *_stats.json in root
    json_paths = []
    for p in root.rglob("*.json"):
        # skip shard stats files like 00037_stats.json (those are in root, not inside shard dir)
        if p.name.endswith("_stats.json"):
            continue
        json_paths.append(p)

    print(f"Found per-image JSON files: {len(json_paths):,} (this should be close to #downloaded images)")

    url2jpg = {}

    # Read and index
    for i, jp in enumerate(json_paths, 1):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        url = extract_url_from_img2dataset_json(obj)
        if not url:
            continue

        # Local jpg path uses same stem as json
        jpg_path = jp.with_suffix(".jpg")
        if jpg_path.exists():
            # First win is fine; duplicates can happen (full/preview/thumb) but mapping url is exact.
            url2jpg[url] = str(jpg_path)

        if i % 20000 == 0:
            print(f"  indexed {i:,}/{len(json_paths):,} json...")

    print(f"URL->JPG entries indexed: {len(url2jpg):,}")
    return url2jpg


print("\nIndexing downloaded images (URL -> local JPG)...")
url2jpg = build_url_to_local_jpg_index(IMAGE_ROOT)

if len(url2jpg) == 0:
    raise RuntimeError(
        "Built an empty URL->JPG index. That means your per-image .json files do not contain URLs "
        "or IMAGE_ROOT is not the correct directory."
    )


# ======================
# STEP 2: ARTIFACT -> LOCAL IMAGES (USING URL JOIN)
# ======================
def get_local_images_for_artifact(artifact_uuid_hex: str, max_images: int | None = None) -> list[str]:
    """
    Uses mapping (artifact_uuid -> image_url) and url2jpg to retrieve local jpg paths.
    """
    rows = mapping[mapping["artifact_uuid"] == artifact_uuid_hex]
    if rows.empty:
        return []

    local = []
    for url in rows["image_url"].tolist():
        p = url2jpg.get(url)
        if p and os.path.exists(p):
            local.append(p)
        if max_images is not None and len(local) >= max_images:
            break
    return local


# ======================
# STEP 3: COLUMN DESCRIPTIONS (FOR SLIDE 1)
# ======================
# Keep this tight, but include everything. For unknown fields: generic explanation.
COLUMN_DESCRIPTIONS = {
    "uuid": "Internal UUID (stored as bytes in parquet). Use uuid_hex for readable/joinable form.",
    "uuid_hex": "Artifact UUID in hex string form. Join key to image_to_artifact_mapping.csv.",
    "label": "Human-readable artifact label/title.",
    "slug": "URL-friendly string identifier.",
    "project_uuid": "Project UUID that the artifact belongs to.",
    "project_label": "Human-readable project name.",
    "context_uuid": "Context UUID (provenance/excavation/context record).",
    "item_class_uuid": "Artifact class/category UUID.",
    "item_uuid": "Linked item UUID for spacetime/geo join (pipeline-specific).",
    "item_uuid_hex": "Hex string for item_uuid.",
    "image_count": "Number of image URLs associated with the artifact (from resources join).",
    "has_location": "Boolean: artifact has usable geographic information after processing.",
    "has_date": "Boolean: artifact has usable temporal information after processing.",
    "latitude": "Latitude in decimal degrees (if available).",
    "longitude": "Longitude in decimal degrees (if available).",
    "geometry_type": "Geometry type (e.g., Point/Polygon) for spatial data.",
    "geometry": "Serialized geometry payload (pipeline-specific; may be WKT/GeoJSON-like).",
    "geo_zoom": "Zoom level / spatial resolution indicator (pipeline-specific).",
    "geo_note": "Notes about geo inference / provenance (pipeline-specific).",
    "geo_depth": "Depth/priority in recursive geo resolution (pipeline-specific).",
    "geo_source_uuid": "UUID of the source used for geo assignment.",
    "geo_spacetime_uuid": "UUID of chosen spacetime geo record.",
    "geo_specificity_x": "Geo specificity score from earlier stage (pipeline-specific).",
    "geo_specificity_y": "Geo specificity score after merge/ranking (pipeline-specific).",
    "geonames_id": "GeoNames identifier if matched.",
    "pleiades_id": "Pleiades identifier if matched.",
    "wikidata_id": "Wikidata identifier if matched.",
    "chrono_depth": "Depth/priority in recursive chrono resolution (pipeline-specific).",
    "chrono_source_uuid": "UUID of the source used for chronological assignment.",
    "chrono_spacetime_uuid": "UUID of chosen spacetime chrono record.",
    "earliest": "Earliest plausible date for the artifact/event (numeric year or year-like).",
    "latest": "Latest plausible date for the artifact/event (numeric year or year-like).",
    "start": "Chosen start of temporal interval (used for coverage/plots).",
    "stop": "Chosen end of temporal interval (used for coverage/plots).",
    "reference_type": "Type of temporal reference (pipeline-specific).",
    "quality_score": (
        "Overall ranking score for spacetime record selection. "
        "Higher = better combined confidence/precision (based on your ranking pipeline)."
    ),
    "is_best": "Boolean: this row is the selected 'best' spacetime candidate for the artifact.",
    # Other ids/flags:
    "edit_group_id": "OpenContext edit grouping identifier.",
    "edit_status": "OpenContext edit/publishing status.",
    "flag_do_not_index": "Boolean: indicates record should not be indexed.",
    "flag_human_remains": "Boolean: indicates sensitive category (human remains).",
    "legacy_id": "Legacy identifier (source-specific).",
    "legacy_source_id": "Legacy source identifier (source-specific).",
    "legacy_duplicate_ids": "Legacy duplicate identifiers, if any.",
    "sitemap_index_id": "Sitemap indexing id (source-specific).",
    "view_group_id": "View grouping id (source-specific).",
    "metadata": "Serialized metadata payload (often JSON-like).",
}

col_desc_path = os.path.join(OUTPUT_DIR, "column_descriptions.txt")
with open(col_desc_path, "w", encoding="utf-8") as f:
    for col in df.columns:
        f.write(f"{col}: {COLUMN_DESCRIPTIONS.get(col, 'Field produced by the OpenContext + spacetime processing pipeline.')}\n")
print(f"Saved: {col_desc_path}")


# ======================
# STEP 4: PICK TWO EXAMPLE ARTIFACTS (WITH REAL LOCAL IMAGES)
# ======================
# Limit to artifacts in df AND in mapping
df_uuids = set(df["uuid_hex"].astype(str).tolist())
mapped_uuids = set(mapping["artifact_uuid"].tolist())
candidate_uuids = list(df_uuids.intersection(mapped_uuids))

if not candidate_uuids:
    raise RuntimeError("No overlap between dataset uuid_hex and mapping artifact_uuid. Check you're using the correct files.")

# Compute local-image counts per artifact (based on URL->JPG availability)
print("\nComputing local image counts per artifact (for picking examples)...")
local_img_counts = {}
for u in random.sample(candidate_uuids, min(5000, len(candidate_uuids))):
    local_img_counts[u] = len(get_local_images_for_artifact(u, max_images=None))

# Prefer:
# - artifact A: many images (>=4)
# - artifact B: fewer images (1-2)
many = [u for u, c in local_img_counts.items() if c >= 4]
few  = [u for u, c in local_img_counts.items() if 1 <= c <= 2]

if not many or not few:
    # fall back to full pass (slower but safe)
    print("Not enough examples found in sample; scanning more candidates...")
    many, few = [], []
    for u in candidate_uuids:
        c = len(get_local_images_for_artifact(u, max_images=None))
        if c >= 4 and not many:
            many.append(u)
        if 1 <= c <= 2 and not few:
            few.append(u)
        if many and few:
            break

if not many or not few:
    raise RuntimeError(
        "Could not find two artifacts with local images (>=4 and 1-2). "
        "This indicates URL->JPG index mismatch with mapping URLs."
    )

artifact1_uuid = many[0]
artifact2_uuid = few[0]
print(f"Selected artifact 1 (many images): {artifact1_uuid}")
print(f"Selected artifact 2 (few images):  {artifact2_uuid}")


# ======================
# STEP 5: EXPORT ARTIFACT CARD TEXT + INDIVIDUAL IMAGES (NO COLLAGE)
# ======================
def write_artifact_card_text(artifact_uuid_hex: str, out_txt: str):
    row = df[df["uuid_hex"] == artifact_uuid_hex]
    if row.empty:
        raise RuntimeError(f"Artifact {artifact_uuid_hex} not found in dataset.")
    row = row.iloc[0]

    with open(out_txt, "w", encoding="utf-8") as f:
        for col in df.columns:
            f.write(f"{col}: {row[col]}\n")


def export_artifact_images_individual(artifact_uuid_hex: str, label: str, max_images: int = 12):
    out_dir = os.path.join(OUTPUT_DIR, f"{label}_images")
    os.makedirs(out_dir, exist_ok=True)

    imgs = get_local_images_for_artifact(artifact_uuid_hex, max_images=max_images)
    if not imgs:
        raise RuntimeError(f"No local images found for {artifact_uuid_hex} even after URL->JPG join.")

    for i, src in enumerate(imgs, 1):
        dst = os.path.join(out_dir, f"{label}_img_{i:03d}.jpg")
        shutil.copy(src, dst)

    print(f"Saved {len(imgs)} images -> {out_dir}")
    return out_dir, imgs


# Artifact card text files
card1_txt = os.path.join(OUTPUT_DIR, "artifact_card_1.txt")
card2_txt = os.path.join(OUTPUT_DIR, "artifact_card_2.txt")
write_artifact_card_text(artifact1_uuid, card1_txt)
write_artifact_card_text(artifact2_uuid, card2_txt)
print(f"Saved: {card1_txt}")
print(f"Saved: {card2_txt}")

# Artifact images (individual)
export_artifact_images_individual(artifact1_uuid, "artifact1", max_images=12)
export_artifact_images_individual(artifact2_uuid, "artifact2", max_images=12)


# ======================
# STEP 6: EXAMPLE IMAGES SLIDE (MONTAGE, OK FOR PPT)
# ======================
# Pull 12 random local jpgs from url2jpg
all_local_jpgs = list(url2jpg.values())
random.shuffle(all_local_jpgs)
example_paths = all_local_jpgs[:12]

# Make a 3x4 montage
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
axes = axes.flatten()

for ax, p in zip(axes, example_paths):
    try:
        img = plt.imread(p)
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")
    ax.axis("off")

for ax in axes[len(example_paths):]:
    ax.axis("off")

plt.tight_layout()
example_out = os.path.join(OUTPUT_DIR, "example_images.png")
plt.savefig(example_out, dpi=250)
plt.close()
print(f"Saved: {example_out}")


# ======================
# STEP 7: DATASET STATISTICS (MORE DETAILED)
# ======================
def safe_numeric_series(s: pd.Series) -> pd.Series:
    # Convert to numeric if possible
    return pd.to_numeric(s, errors="coerce")

# Core counts
n_artifacts = len(df)

# Images (from mapping + local availability)
artifacts_in_mapping = df["uuid_hex"].isin(mapping["artifact_uuid"]).sum()
unique_artifacts_in_mapping = len(set(df["uuid_hex"]).intersection(set(mapping["artifact_uuid"])))

# Per-artifact URL counts (from mapping)
url_count_per_artifact = mapping.groupby("artifact_uuid")["image_url"].nunique()

# Per-artifact LOCAL counts (subset; compute for intersection only to keep cost reasonable)
# We'll compute exact for artifacts present in mapping and dataset, but that could be large; it’s still fine at ~33k.
print("\nComputing local image counts for ALL artifacts in the final dataset (this may take a minute)...")
local_count_per_artifact = {}
for u in df["uuid_hex"].astype(str).tolist():
    # Only check if artifact has any urls
    if u in mapped_uuids:
        local_count_per_artifact[u] = len(get_local_images_for_artifact(u, max_images=None))
    else:
        local_count_per_artifact[u] = 0
local_counts = pd.Series(local_count_per_artifact)

# Location/date coverage
has_loc = df["has_location"].sum() if "has_location" in df.columns else None
has_date = df["has_date"].sum() if "has_date" in df.columns else None
lat_nonnull = df["latitude"].notna().sum() if "latitude" in df.columns else None
lon_nonnull = df["longitude"].notna().sum() if "longitude" in df.columns else None

# Temporal
start_num = safe_numeric_series(df["start"]) if "start" in df.columns else pd.Series([], dtype=float)
stop_num = safe_numeric_series(df["stop"]) if "stop" in df.columns else pd.Series([], dtype=float)

# Quality
quality = safe_numeric_series(df["quality_score"]) if "quality_score" in df.columns else pd.Series([], dtype=float)

stats_lines = []
stats_lines.append(f"TOTAL ARTIFACTS (rows in final dataset): {n_artifacts:,}")
stats_lines.append("")
stats_lines.append("IMAGES")
stats_lines.append(f"- Artifacts with >=1 image URL (from mapping join): {unique_artifacts_in_mapping:,}")
stats_lines.append(f"- Total unique image URLs in mapping: {mapping['image_url'].nunique():,}")
stats_lines.append(f"- Total downloaded local JPGs indexed: {len(url2jpg):,}")
stats_lines.append(f"- Local images per artifact (based on URL->JPG match):")
stats_lines.append(f"    mean={local_counts.mean():.2f}  median={local_counts.median():.0f}  max={local_counts.max():.0f}")
stats_lines.append(f"    artifacts with 0 local imgs: {(local_counts==0).sum():,}")
stats_lines.append(f"    artifacts with 1-2 local imgs: ((local_counts>=1)&(local_counts<=2)).sum() = {(((local_counts>=1)&(local_counts<=2)).sum()):,}")
stats_lines.append(f"    artifacts with >=3 local imgs: {(local_counts>=3).sum():,}")
stats_lines.append("")
stats_lines.append("SPATIAL")
if has_loc is not None:
    stats_lines.append(f"- has_location=True: {has_loc:,} ({has_loc/n_artifacts*100:.1f}%)")
if lat_nonnull is not None and lon_nonnull is not None:
    both = int(((df["latitude"].notna()) & (df["longitude"].notna())).sum())
    stats_lines.append(f"- Has (lat,lon) both present: {both:,} ({both/n_artifacts*100:.1f}%)")
stats_lines.append("")
stats_lines.append("TEMPORAL")
if has_date is not None:
    stats_lines.append(f"- has_date=True: {has_date:,} ({has_date/n_artifacts*100:.1f}%)")
if len(start_num.dropna()) > 0 and len(stop_num.dropna()) > 0:
    stats_lines.append(f"- start year range: [{start_num.min():.0f}, {start_num.max():.0f}]")
    stats_lines.append(f"- stop  year range: [{stop_num.min():.0f}, {stop_num.max():.0f}]")
stats_lines.append("")
stats_lines.append("QUALITY SCORE")
if len(quality.dropna()) > 0:
    desc = quality.describe()
    stats_lines.append(str(desc))
stats_lines.append("")
stats_lines.append(f"EXAMPLE ARTIFACTS USED IN SLIDES")
stats_lines.append(f"- Artifact 1 uuid_hex: {artifact1_uuid}")
stats_lines.append(f"- Artifact 2 uuid_hex: {artifact2_uuid}")

stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stats_lines))
print(f"Saved: {stats_path}")


# ======================
# STEP 8: COVERAGE DIAGRAMS
# ======================
print("\nGenerating coverage diagrams...")

# Geo scatter
if "longitude" in df.columns and "latitude" in df.columns:
    g = df[df["longitude"].notna() & df["latitude"].notna()]
    plt.figure(figsize=(10, 7))
    plt.scatter(g["longitude"], g["latitude"], s=1, alpha=0.25)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Coverage (lat/lon points)")
    geo_out = os.path.join(OUTPUT_DIR, "geo_coverage.png")
    plt.savefig(geo_out, dpi=250)
    plt.close()
    print(f"Saved: {geo_out}")

# Temporal histogram (start)
if "start" in df.columns:
    s = safe_numeric_series(df["start"]).dropna()
    if len(s) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(s, bins=80)
        plt.xlabel("Start year")
        plt.ylabel("Count")
        plt.title("Temporal Coverage (start year distribution)")
        t_out = os.path.join(OUTPUT_DIR, "temporal_coverage.png")
        plt.savefig(t_out, dpi=250)
        plt.close()
        print(f"Saved: {t_out}")

# Quality histogram
if "quality_score" in df.columns:
    q = safe_numeric_series(df["quality_score"]).dropna()
    if len(q) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(q, bins=60)
        plt.xlabel("quality_score")
        plt.ylabel("Count")
        plt.title("Quality Score Distribution")
        q_out = os.path.join(OUTPUT_DIR, "quality_distribution.png")
        plt.savefig(q_out, dpi=250)
        plt.close()
        print(f"Saved: {q_out}")

# Images-per-artifact histogram (LOCAL)
plt.figure(figsize=(10, 6))
plt.hist(local_counts.values, bins=40)
plt.xlabel("Local images per artifact")
plt.ylabel("Count")
plt.title("Downloaded Local Images per Artifact (URL->JPG matched)")
ipc_out = os.path.join(OUTPUT_DIR, "images_per_artifact.png")
plt.savefig(ipc_out, dpi=250)
plt.close()
print(f"Saved: {ipc_out}")


print("\nDONE ✅")
print(f"All outputs are in: {OUTPUT_DIR}")
print("\nKey files for your slides:")
print(f"  1) {OUTPUT_DIR}/column_descriptions.txt")
print(f"  2) {OUTPUT_DIR}/example_images.png")
print(f"  3) {OUTPUT_DIR}/artifact_card_1.txt  + {OUTPUT_DIR}/artifact1_images/")
print(f"  4) {OUTPUT_DIR}/artifact_card_2.txt  + {OUTPUT_DIR}/artifact2_images/")
print(f"  5) {OUTPUT_DIR}/dataset_statistics.txt")
print(f"  6) {OUTPUT_DIR}/geo_coverage.png, temporal_coverage.png, quality_distribution.png, images_per_artifact.png")
