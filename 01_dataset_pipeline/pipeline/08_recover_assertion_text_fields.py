#!/usr/bin/env python3
import json
import re
import html
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/home/udedhia/archaia_project/data"
BASE_DATASET = "/data/group_data/dei-group/archaia/archaia_final_dataset.parquet"
OUT_PREFIX = "/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1"

ASSERTIONS_PATH = f"{DATA_DIR}/oc_all_assertions.parquet"
MANIFEST_PATH = f"{DATA_DIR}/oc_all_manifest.parquet"

NULL_OBJECT_LABELS = {"Default (Null) Attribute Object"}
NULL_OBJECT_SLUGS = {"oc-default-object-null"}

# Predicates worth scanning into the semantic-text layer
KEEP_RE = re.compile(
    r"(artifact name|material|munsell|color|description|new description|has note|"
    r"object type|period|chronotype|fabric|decorative|size|specific context|specific location|"
    r"location|locus|function|condition|registration date|disposition)",
    re.IGNORECASE
)

# Exact dedicated field mapping
TARGET_FIELDS = {
    "artifact name": "recovered_artifact_name",
    "material": "recovered_material",
    "material (note)": "recovered_material_note",
    "has note": "recovered_note",
    "description": "recovered_description",
    "new description": "recovered_description",
    "description remarks": "recovered_description_remarks",
    "object type": "recovered_object_type",
    "object type (notes)": "recovered_object_type_note",
    "period": "recovered_period",
    "chronotype": "recovered_chronotype",
    "fabric description": "recovered_fabric_description",
    "fabric group": "recovered_fabric_group",
    "munsell color": "recovered_munsell_color",
    "munsell #": "recovered_munsell_number",
    "decorative technique": "recovered_decorative_technique",
    "size": "recovered_size",
    "specific context": "recovered_specific_context",
    "specific location": "recovered_specific_location",
    "location": "recovered_location",
    "locus": "recovered_locus",
    "locus id": "recovered_locus_id",
    "function": "recovered_function",
    "condition": "recovered_condition",
    "registration date": "recovered_registration_date",
    "disposition": "recovered_disposition",
}

# ============================================================
# HELPERS
# ============================================================
def to_hex(x):
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    return str(x)

def stringify_primitive(v):
    if pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)

def strip_html(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text)
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_predicate_label(label: str) -> str:
    if label is None:
        return ""
    s = str(label).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def resolved_value(object_label, object_slug, obj_string, obj_datetime, obj_integer, obj_double, obj_boolean):
    # Prefer object label only if it's not the placeholder null object
    if pd.notna(object_label):
        olab = str(object_label).strip()
        oslug = str(object_slug).strip() if pd.notna(object_slug) else ""
        if olab not in NULL_OBJECT_LABELS and oslug not in NULL_OBJECT_SLUGS:
            return strip_html(olab)

    # Otherwise use primitive value columns
    for v in [obj_string, obj_datetime, obj_integer, obj_double, obj_boolean]:
        sv = stringify_primitive(v)
        if sv is not None:
            return strip_html(sv)

    return None

def join_unique(values):
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    if not out:
        return None
    return " || ".join(out)

# ============================================================
# LOAD BASE DATASET
# ============================================================
print("Loading base final dataset...")
base = pd.read_parquet(BASE_DATASET)
target_hex = set(base["uuid_hex"].astype(str).tolist())
print(f"Target artifacts in base dataset: {len(target_hex):,}")

# ============================================================
# LOAD MANIFEST LOOKUPS
# ============================================================
print("Loading manifest lookups...")
manifest = pd.read_parquet(
    MANIFEST_PATH,
    columns=["uuid", "label", "slug", "item_type", "data_type"]
)
manifest["uuid_hex"] = manifest["uuid"].apply(to_hex)

lookup = {}
for _, r in manifest.iterrows():
    lookup[r["uuid"]] = {
        "uuid_hex": r["uuid_hex"],
        "label": r["label"],
        "slug": r["slug"],
        "item_type": r["item_type"],
        "data_type": r["data_type"],
    }

# ============================================================
# SCAN ASSERTIONS ROW-GROUP BY ROW-GROUP
# ============================================================
print("Scanning assertions row groups...")
pf = pq.ParquetFile(ASSERTIONS_PATH)

needed_cols = [
    "subject_uuid", "predicate_uuid", "object_uuid",
    "obj_string", "obj_boolean", "obj_integer", "obj_double", "obj_datetime"
]

long_rows = []

for i in range(pf.num_row_groups):
    table = pf.read_row_group(i, columns=needed_cols)
    df = table.to_pandas()

    df["subject_uuid_hex"] = df["subject_uuid"].apply(to_hex)
    df = df[df["subject_uuid_hex"].isin(target_hex)].copy()

    if len(df) == 0:
        continue

    print(f"  row group {i}: kept {len(df):,} rows")

    for _, r in df.iterrows():
        pred = lookup.get(r["predicate_uuid"], {})
        obj = lookup.get(r["object_uuid"], {}) if pd.notna(r.get("object_uuid")) else {}

        predicate_label = pred.get("label")
        if not predicate_label or not KEEP_RE.search(str(predicate_label)):
            continue

        predicate_slug = pred.get("slug")
        predicate_data_type = pred.get("data_type")

        object_label = obj.get("label")
        object_slug = obj.get("slug")

        value = resolved_value(
            object_label=object_label,
            object_slug=object_slug,
            obj_string=r.get("obj_string"),
            obj_datetime=r.get("obj_datetime"),
            obj_integer=r.get("obj_integer"),
            obj_double=r.get("obj_double"),
            obj_boolean=r.get("obj_boolean"),
        )

        if value is None:
            continue

        pred_norm = normalize_predicate_label(predicate_label)

        long_rows.append({
            "uuid_hex": r["subject_uuid_hex"],
            "predicate_label": predicate_label,
            "predicate_slug": predicate_slug,
            "predicate_data_type": predicate_data_type,
            "object_label": object_label,
            "object_slug": object_slug,
            "resolved_value": value,
            "normalized_predicate_label": pred_norm,
        })

# ============================================================
# LONG-FORM OUTPUT
# ============================================================
print("Building long-form recovered assertions...")
long_df = pd.DataFrame(long_rows)

long_out_csv = f"{OUT_PREFIX}__recovered_assertions_long.csv"
long_df.to_csv(long_out_csv, index=False)
print(f"Wrote: {long_out_csv}")

# ============================================================
# BUILD WIDE PER-ARTIFACT RECOVERY
# ============================================================
print("Building per-artifact recovered fields...")

all_fields = defaultdict(lambda: defaultdict(list))
dedicated_values = defaultdict(lambda: defaultdict(list))

for _, r in long_df.iterrows():
    uuid_hex = r["uuid_hex"]
    pred_label = r["predicate_label"]
    pred_norm = r["normalized_predicate_label"]
    value = r["resolved_value"]

    all_fields[uuid_hex][pred_label].append(value)

    if pred_norm in TARGET_FIELDS:
        col = TARGET_FIELDS[pred_norm]
        dedicated_values[uuid_hex][col].append(value)

wide_rows = []
for uuid_hex in target_hex:
    row = {"uuid_hex": uuid_hex}

    # dedicated columns
    for pred_norm, col in TARGET_FIELDS.items():
        row[col] = join_unique(dedicated_values[uuid_hex].get(col, []))

    # catch-all JSON
    payload = {}
    for k, vals in all_fields.get(uuid_hex, {}).items():
        cleaned = [v for v in vals if v is not None]
        uniq = []
        seen = set()
        for v in cleaned:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        if uniq:
            payload[k] = uniq

    row["recovered_text_fields_json"] = json.dumps(payload, ensure_ascii=False) if payload else None
    wide_rows.append(row)

wide_df = pd.DataFrame(wide_rows)

# ============================================================
# MERGE ONTO BASE DATASET
# ============================================================
print("Merging onto base dataset...")
aug = base.merge(wide_df, on="uuid_hex", how="left")

def build_search_text(row):
    parts = []

    ordered_cols = [
        "label",
        "project_label",
        "recovered_artifact_name",
        "recovered_material",
        "recovered_material_note",
        "recovered_object_type",
        "recovered_object_type_note",
        "recovered_period",
        "recovered_chronotype",
        "recovered_fabric_description",
        "recovered_fabric_group",
        "recovered_munsell_color",
        "recovered_munsell_number",
        "recovered_decorative_technique",
        "recovered_size",
        "recovered_specific_context",
        "recovered_specific_location",
        "recovered_location",
        "recovered_locus",
        "recovered_locus_id",
        "recovered_function",
        "recovered_condition",
        "recovered_description",
        "recovered_description_remarks",
        "recovered_note",
        "recovered_registration_date",
        "recovered_disposition",
    ]

    for col in ordered_cols:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}: {row[col]}")

    return "\n".join(parts) if parts else None

aug["recovered_search_text_v1"] = aug.apply(build_search_text, axis=1)

# ============================================================
# SAVE NEW DATASET ONLY
# ============================================================
out_parquet = f"{OUT_PREFIX}.parquet"
out_csv = f"{OUT_PREFIX}.csv"

aug.to_parquet(out_parquet, index=False)
aug.to_csv(out_csv, index=False)

print(f"Wrote: {out_parquet}")
print(f"Wrote: {out_csv}")

# ============================================================
# COVERAGE SUMMARY
# ============================================================
print("\nCoverage summary:")
for col in [
    "recovered_artifact_name",
    "recovered_material",
    "recovered_material_note",
    "recovered_note",
    "recovered_description",
    "recovered_object_type",
    "recovered_object_type_note",
    "recovered_period",
    "recovered_chronotype",
    "recovered_fabric_description",
    "recovered_fabric_group",
    "recovered_munsell_color",
    "recovered_munsell_number",
    "recovered_decorative_technique",
    "recovered_size",
    "recovered_specific_context",
    "recovered_specific_location",
    "recovered_location",
    "recovered_locus",
    "recovered_locus_id",
    "recovered_function",
    "recovered_condition",
    "recovered_registration_date",
    "recovered_disposition",
    "recovered_text_fields_json",
]:
    nonnull = aug[col].notna().sum()
    print(f"{col:30s} {nonnull:8d} / {len(aug):8d} ({nonnull * 100 / len(aug):6.2f}%)")
