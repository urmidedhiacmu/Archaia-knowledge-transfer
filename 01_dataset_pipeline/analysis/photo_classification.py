import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

MANIFEST_PATH = "../data/oc_all_manifest.parquet"
ASSERTIONS_PATH = "../data/oc_all_assertions.parquet"

OUTPUT_DIR = Path("/data/group_data/dei-group/archaia/oc_media_classification_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# If auto-detection fails, manually set these after inspection
ASSERTION_LEFT_UUID_COL = "subject_uuid"
ASSERTION_RIGHT_UUID_COL = "object_uuid"

# Optional exact labels; leaving empty uses fuzzy matching
KNOWN_MEDIA_CLASS_LABELS = set()

# =========================================================
# HELPERS
# =========================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def safe_str_lower(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def normalize_uuid_value(x):
    """
    Convert UUID-like values to a safe comparable string.
    Handles bytes/memoryview by converting to hex.
    """
    if x is None:
        return np.nan

    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    if isinstance(x, memoryview):
        x = x.tobytes()

    if isinstance(x, (bytes, bytearray)):
        return x.hex()

    if isinstance(x, str):
        return x.strip()

    return str(x)

def normalize_uuid_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_uuid_value)

def infer_depicts_type(linked_class: str) -> str:
    x = safe_str_lower(linked_class)
    if "object" in x or "artifact" in x or "pottery" in x or "lithic" in x or "bone" in x or "coin" in x:
        return "artifact_photo"
    if "locus" in x:
        return "locus_photo"
    if "site" in x:
        return "site_photo"
    if "feature" in x:
        return "feature_photo"
    if "unit" in x or "excavation unit" in x or "survey unit" in x:
        return "unit_photo"
    if "trench" in x:
        return "trench_photo"
    if "structure" in x:
        return "structure_photo"
    if x == "":
        return "unknown"
    return "other"

def detect_uuid_columns(assertions_df: pd.DataFrame, manifest_uuid_set: set):
    """
    Score UUID-like columns in assertions by overlap with manifest UUIDs.
    """
    candidate_cols = [c for c in assertions_df.columns if "uuid" in c.lower()]
    scores = []

    for col in candidate_cols:
        try:
            norm = normalize_uuid_series(assertions_df[col].head(50000))
            overlap = norm.isin(manifest_uuid_set).sum()
            frac = overlap / max(len(norm), 1)
            scores.append((col, overlap, frac))
        except Exception as e:
            print(f"Skipping {col} during UUID detection due to error: {e}")

    scores = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)
    return scores

def summarize_unique_classes(manifest_with_class: pd.DataFrame):
    vc = (
        manifest_with_class["item_class_label"]
        .fillna("<<MISSING>>")
        .value_counts(dropna=False)
        .reset_index()
    )
    vc.columns = ["item_class_label", "count"]
    return vc

def identify_media_rows(manifest_with_class: pd.DataFrame):
    cls = manifest_with_class["item_class_label"].fillna("").astype(str)

    if KNOWN_MEDIA_CLASS_LABELS:
        media_mask = cls.str.lower().isin({x.lower() for x in KNOWN_MEDIA_CLASS_LABELS})
    else:
        media_mask = cls.str.lower().str.contains("media|image|photo|photograph|document media", regex=True)

    return manifest_with_class[media_mask].copy()

def choose_best_depicts(group: pd.Series) -> str:
    vals = set(group.dropna().astype(str))
    priority = [
        "artifact_photo",
        "locus_photo",
        "site_photo",
        "feature_photo",
        "unit_photo",
        "trench_photo",
        "structure_photo",
        "other",
        "unknown",
    ]
    for p in priority:
        if p in vals:
            return p
    return "unknown"

# =========================================================
# LOAD
# =========================================================

print_header("LOADING PARQUETS")
manifest = pd.read_parquet(MANIFEST_PATH)
assertions = pd.read_parquet(ASSERTIONS_PATH)

print(f"Manifest shape:   {manifest.shape}")
print(f"Assertions shape: {assertions.shape}")

print_header("MANIFEST COLUMNS")
print(manifest.columns.tolist())

print_header("ASSERTIONS COLUMNS")
print(assertions.columns.tolist())

# =========================================================
# NORMALIZE MANIFEST UUIDS
# =========================================================

print_header("NORMALIZING MANIFEST UUID FIELDS")

manifest["uuid_norm"] = normalize_uuid_series(manifest["uuid"])
manifest["item_class_uuid_norm"] = normalize_uuid_series(manifest["item_class_uuid"])

for optional_col in ["context_uuid", "project_uuid", "publisher_uuid"]:
    if optional_col in manifest.columns:
        manifest[f"{optional_col}_norm"] = normalize_uuid_series(manifest[optional_col])

# =========================================================
# DECODE ITEM CLASS
# =========================================================

print_header("DECODING item_class_uuid VIA SELF-JOIN")

required_manifest_cols = {"uuid_norm", "label", "item_class_uuid_norm"}
missing_manifest_cols = required_manifest_cols - set(manifest.columns)
if missing_manifest_cols:
    raise ValueError(f"Manifest missing required columns: {missing_manifest_cols}")

class_lookup = manifest[["uuid_norm", "label"]].rename(
    columns={
        "uuid_norm": "item_class_uuid_norm",
        "label": "item_class_label",
    }
)

manifest_with_class = manifest.merge(
    class_lookup,
    on="item_class_uuid_norm",
    how="left",
)

manifest_with_class_path = OUTPUT_DIR / "manifest_with_item_class.parquet"
manifest_with_class.to_parquet(manifest_with_class_path, index=False)
print(f"Saved: {manifest_with_class_path}")

# =========================================================
# CLASS LABEL INSPECTION
# =========================================================

print_header("TOP ITEM CLASS LABELS")
class_counts = summarize_unique_classes(manifest_with_class)
print(class_counts.head(50).to_string(index=False))

class_counts_path = OUTPUT_DIR / "item_class_counts.csv"
class_counts.to_csv(class_counts_path, index=False)
print(f"Saved: {class_counts_path}")

# =========================================================
# IDENTIFY MEDIA RECORDS
# =========================================================

print_header("IDENTIFYING MEDIA RECORDS")

media_records = identify_media_rows(manifest_with_class)
print(f"Detected media records: {len(media_records)}")

media_records_path = OUTPUT_DIR / "media_records.parquet"
media_records.to_parquet(media_records_path, index=False)
print(f"Saved: {media_records_path}")

if len(media_records) == 0:
    print("No media rows found. Check item_class_counts.csv.")
    raise SystemExit(0)

media_records["uuid_norm"] = normalize_uuid_series(media_records["uuid"])
media_ids = set(media_records["uuid_norm"].dropna())
manifest_uuid_set = set(manifest_with_class["uuid_norm"].dropna())

# =========================================================
# DETECT ASSERTION UUID COLUMNS
# =========================================================

print_header("USING ASSERTION UUID COLUMNS")

ASSERTION_LEFT_UUID_COL = "subject_uuid"
ASSERTION_RIGHT_UUID_COL = "object_uuid"

print(f"Using assertion UUID columns:")
print(f"  LEFT  = {ASSERTION_LEFT_UUID_COL}")
print(f"  RIGHT = {ASSERTION_RIGHT_UUID_COL}")

left_col = ASSERTION_LEFT_UUID_COL
right_col = ASSERTION_RIGHT_UUID_COL

assertions[left_col + "_norm"] = normalize_uuid_series(assertions[left_col])
assertions[right_col + "_norm"] = normalize_uuid_series(assertions[right_col])

# =========================================================
# TRACE MEDIA LINKS IN BOTH DIRECTIONS
# =========================================================

print_header("TRACING MEDIA RECORDS THROUGH ASSERTIONS")

# Case 1: media is subject, linked record is object
a_left_media = assertions[assertions[left_col + "_norm"].isin(media_ids)].copy()
a_left_media["media_uuid"] = a_left_media[left_col + "_norm"]
a_left_media["linked_uuid"] = a_left_media[right_col + "_norm"]
a_left_media["relation_direction"] = f"{left_col} -> {right_col}"

# Case 2: media is object, linked record is subject
a_right_media = assertions[assertions[right_col + "_norm"].isin(media_ids)].copy()
a_right_media["media_uuid"] = a_right_media[right_col + "_norm"]
a_right_media["linked_uuid"] = a_right_media[left_col + "_norm"]
a_right_media["relation_direction"] = f"{right_col} -> {left_col}"

media_links_raw = pd.concat([a_left_media, a_right_media], ignore_index=True)

print(f"Links found with media on LEFT side:  {len(a_left_media)}")
print(f"Links found with media on RIGHT side: {len(a_right_media)}")
print(f"Total raw media links:                {len(media_links_raw)}")

if "predicate_uuid" in media_links.columns:
    pred_lookup = manifest_with_class[["uuid_norm", "label"]].rename(
        columns={"uuid_norm": "predicate_uuid_norm", "label": "predicate_label"}
    )
    media_links["predicate_uuid_norm"] = normalize_uuid_series(media_links["predicate_uuid"])
    media_links = media_links.merge(pred_lookup, on="predicate_uuid_norm", how="left")

    print_header("TOP PREDICATES USED IN MEDIA LINKS")
    print(media_links["predicate_label"].fillna("<<MISSING>>").value_counts().head(20).to_string())
    
# keep only links to known manifest entities
media_links_raw = media_links_raw[media_links_raw["linked_uuid"].isin(manifest_uuid_set)].copy()

# remove self-links if any
media_links_raw = media_links_raw[media_links_raw["media_uuid"] != media_links_raw["linked_uuid"]].copy()

print(f"After filtering to valid linked manifest UUIDs: {len(media_links_raw)}")

if len(media_links_raw) == 0:
    print("Still no media links found.")
    raise SystemExit(0)
# =========================================================
# ATTACH MEDIA + LINKED RECORD INFO
# =========================================================

print_header("ATTACHING LABELS AND CLASSES TO MEDIA LINKS")

media_info = manifest_with_class[["uuid_norm", "label", "item_class_label"]].rename(
    columns={
        "uuid_norm": "media_uuid",
        "label": "media_label",
        "item_class_label": "media_class",
    }
)

linked_info = manifest_with_class[["uuid_norm", "label", "item_class_label"]].rename(
    columns={
        "uuid_norm": "linked_uuid",
        "label": "linked_label",
        "item_class_label": "linked_class",
    }
)

media_links = media_links_raw.merge(media_info, on="media_uuid", how="left")
media_links = media_links.merge(linked_info, on="linked_uuid", how="left")
media_links = media_links[media_links["media_uuid"] != media_links["linked_uuid"]].copy()


# =========================================================
# INFER WHAT EACH MEDIA RECORD DEPICTS
# =========================================================

print_header("INFERRING PHOTO TYPE FROM LINKED RECORD CLASS")

media_links["depicts_type"] = media_links["linked_class"].apply(infer_depicts_type)

print(media_links["depicts_type"].value_counts(dropna=False).to_string())

media_links_path = OUTPUT_DIR / "media_links_with_target_class.parquet"
media_links.to_parquet(media_links_path, index=False)
print(f"Saved: {media_links_path}")

media_links_csv_path = OUTPUT_DIR / "media_links_with_target_class.csv"
media_links.to_csv(media_links_csv_path, index=False)
print(f"Saved: {media_links_csv_path}")

# =========================================================
# COLLAPSE TO ONE ROW PER MEDIA
# =========================================================

print_header("CREATING ONE-LABEL-PER-MEDIA SUMMARY")

media_summary = (
    media_links.groupby("media_uuid")
    .agg(
        media_label=("media_label", "first"),
        media_class=("media_class", "first"),
        num_linked_records=("linked_uuid", "nunique"),
        linked_classes=("linked_class", lambda x: sorted(set([str(v) for v in x.dropna()]))),
        depicts_type=("depicts_type", choose_best_depicts),
    )
    .reset_index()
)

media_summary_path = OUTPUT_DIR / "media_summary_one_label_per_media.parquet"
media_summary.to_parquet(media_summary_path, index=False)
print(f"Saved: {media_summary_path}")

media_summary_csv_path = OUTPUT_DIR / "media_summary_one_label_per_media.csv"
media_summary.to_csv(media_summary_csv_path, index=False)
print(f"Saved: {media_summary_csv_path}")

print(media_summary.head(20).to_string(index=False))

# =========================================================
# FINAL MANIFEST WITH RESOLVED TYPE
# =========================================================

print_header("BUILDING FINAL MANIFEST WITH RESOLVED TYPE")

final_manifest = manifest_with_class.copy()
final_manifest["resolved_record_type"] = final_manifest["item_class_label"]

media_summary_small = media_summary[["media_uuid", "depicts_type"]].rename(
    columns={"media_uuid": "uuid_norm", "depicts_type": "media_depicts_type"}
)

final_manifest = final_manifest.merge(media_summary_small, on="uuid_norm", how="left")

media_like_mask = (
    final_manifest["item_class_label"]
    .fillna("")
    .str.lower()
    .str.contains("media|image|photo|photograph|document media", regex=True)
)

final_manifest.loc[
    media_like_mask & final_manifest["media_depicts_type"].notna(),
    "resolved_record_type"
] = final_manifest.loc[
    media_like_mask & final_manifest["media_depicts_type"].notna(),
    "media_depicts_type"
]

final_manifest_path = OUTPUT_DIR / "final_manifest_with_resolved_record_type.parquet"
final_manifest.to_parquet(final_manifest_path, index=False)
print(f"Saved: {final_manifest_path}")

final_manifest_csv_path = OUTPUT_DIR / "final_manifest_with_resolved_record_type.csv"
final_manifest.to_csv(final_manifest_csv_path, index=False)
print(f"Saved: {final_manifest_csv_path}")

print_header("FINAL RESOLVED TYPE COUNTS")
print(
    final_manifest["resolved_record_type"]
    .fillna("<<MISSING>>")
    .value_counts(dropna=False)
    .head(50)
    .to_string()
)

# =========================================================
# MEETING SUMMARY
# =========================================================

print_header("MEETING SUMMARY")
print(
    f"""
1. Manifest rows were classified by decoding item_class_uuid via self-join.
2. Media rows were identified from classes like 'Image media' and 'Document media'.
3. Assertion UUID columns were normalized from binary/bytes to comparable hex strings.
4. Media records were traced through assertions to linked manifest records.
5. Linked record classes were used to infer whether the media is of an artifact, locus, site, unit, feature, etc.

Outputs written to:
{OUTPUT_DIR}
"""
)

print_header("DONE")