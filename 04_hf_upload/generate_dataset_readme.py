#!/usr/bin/env python3

import pandas as pd

DATA = "/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet"
OUT = "/data/group_data/dei-group/archaia/AUGMENTED_DATASET_COLUMNS.md"

df = pd.read_parquet(DATA)

DESCRIPTIONS = {

# artifact
"label": "Artifact label or catalog identifier used by the excavation project.",
"metadata": "Additional metadata from the source project.",
"project_label": "Name of the archaeological project.",
"slug": "URL-friendly identifier used by OpenContext for the artifact.",

# reference scoring
"is_best": "Whether this record was selected as the best spatial/temporal reference.",
"quality_score": "Score used when selecting the best reference record.",
"reference_type": "Type of reference used for location or chronology.",

# provenance
"context_uuid": "UUID of the archaeological context.",
"item_class_uuid": "UUID identifying the artifact class.",
"project_uuid": "UUID of the project in OpenContext.",

# geographic
"geo_depth": "Depth of spatial inheritance used when assigning location.",
"geo_note": "Notes related to geographic assignment.",
"geo_specificity_y": "Specificity ranking of the geographic reference.",
"geo_zoom": "Zoom level used when deriving geographic coordinates.",
"geometry": "GeoJSON geometry describing artifact location.",
"geometry_type": "Geometry type (typically Point).",
"latitude": "Latitude of artifact location.",
"longitude": "Longitude of artifact location.",

# media
"image_count_y": "Number of images linked to the artifact.",
"image_paths": "Local file paths of downloaded artifact images.",

# temporal
"chrono_depth": "Depth of chronological inheritance.",
"earliest": "Earliest possible date associated with artifact.",
"latest": "Latest possible date associated with artifact.",
"start": "Start date of artifact's temporal range.",
"stop": "End date of artifact's temporal range.",

# recovered fields
"recovered_artifact_name": "Artifact name recovered from OpenContext assertions.",
"recovered_material": "Material of the artifact recovered from assertions.",
"recovered_material_note": "Additional notes about artifact material.",
"recovered_note": "Recovered general note text.",
"recovered_description": "Recovered main description text.",
"recovered_description_remarks": "Recovered extra remarks related to the description.",
"recovered_object_type": "Recovered object type or category.",
"recovered_object_type_note": "Recovered note about object type.",
"recovered_period": "Recovered named archaeological period.",
"recovered_chronotype": "Recovered chronotype classification used by the source project.",
"recovered_fabric_description": "Recovered description of ceramic or material fabric.",
"recovered_fabric_group": "Recovered fabric group classification.",
"recovered_munsell_color": "Recovered Munsell color text.",
"recovered_munsell_number": "Recovered Munsell numeric/code value.",
"recovered_decorative_technique": "Recovered decorative technique field.",
"recovered_size": "Recovered size or measurement text.",
"recovered_specific_context": "Recovered specific archaeological context.",
"recovered_specific_location": "Recovered specific location within the site.",
"recovered_location": "Recovered general location description.",
"recovered_locus": "Recovered excavation locus value.",
"recovered_locus_id": "Recovered locus identifier.",
"recovered_function": "Recovered interpreted function.",
"recovered_condition": "Recovered preservation or condition text.",
"recovered_registration_date": "Recovered registration or cataloguing date.",
"recovered_disposition": "Recovered storage, museum, or disposition field.",
"recovered_text_fields_json": "JSON object containing all recovered assertion text fields for the artifact.",
"recovered_search_text_v1": "Combined text field built from selected recovered fields for search or embeddings.",
}

rows = len(df)

orig_cols = [
    "label", "metadata", "project_label", "slug",
    "is_best", "quality_score", "reference_type",
    "context_uuid", "item_class_uuid", "project_uuid",
    "geo_depth", "geo_note", "geo_specificity_y", "geo_zoom",
    "geometry", "geometry_type", "latitude", "longitude",
    "image_count_y", "image_paths",
    "chrono_depth", "earliest", "latest", "start", "stop",
]

recovered_cols = [c for c in df.columns if c.startswith("recovered_")]

lines = []

# header
lines.append("# Archaia Dataset Columns")
lines.append("")
lines.append("## Overview")
lines.append("")
lines.append("This is the cleaned artifact-level dataset with recovered text metadata added from the OpenContext assertion layer.")
lines.append("")
lines.append(f"- Rows: **{len(df):,}**")
lines.append(f"- Columns: **{len(df.columns):,}**")
lines.append("")
lines.append("The original dataset already contained the artifact rows, project/source information, spatial fields, temporal fields, and image paths.")
lines.append("The recovered fields were added later by scanning OpenContext assertions and pulling back descriptive text fields that were missing.")
lines.append("")

lines.append("## High-level pipeline")
lines.append("")
lines.append("1. Start from the cleaned final artifact dataset (`archaia_final_dataset`).")
lines.append("2. Use artifact UUIDs from that dataset to scan the OpenContext assertions parquet.")
lines.append("3. Resolve assertion values using manifest lookups:")
lines.append("   - use object labels when the value is stored as a linked object")
lines.append("   - use primitive assertion fields (`obj_string`, `obj_datetime`, etc.) when the object is just a null placeholder")
lines.append("4. Keep useful descriptive predicates such as material, note, description, object type, size, period, condition, etc.")
lines.append("5. Add selected recovered values back as separate `recovered_*` columns.")
lines.append("6. Store the full recovered assertion text for each artifact in `recovered_text_fields_json`.")
lines.append("")

lines.append("## Recovered text fields")
lines.append("")
lines.append("The `recovered_*` columns are individual fields pulled out from assertions, for example recovered material, note, description, object type, and condition.")
lines.append("Coverage differs by field because different source projects used different schemas and recorded different kinds of metadata.")
lines.append("")

lines.append("### `recovered_text_fields_json`")
lines.append("")
lines.append("This is the full recovered text metadata for an artifact stored as a JSON object.")
lines.append("It keeps the original assertion field names as keys and stores values as lists, since an artifact can have more than one value for the same field.")
lines.append("")
lines.append("Example shape:")
lines.append("")
lines.append("```json")
lines.append('{')
lines.append('  "Material": ["Obsidian"],')
lines.append('  "Artifact Name": ["Mirror"],')
lines.append('  "Has note": ["Fragment of a very well polished obsidian piece..."]')
lines.append('}')
lines.append("```")
lines.append("")
lines.append("Use this column when you want the most complete recovered metadata without losing field names.")
lines.append("")

if "recovered_search_text_v1" in df.columns:
    lines.append("### `recovered_search_text_v1`")
    lines.append("")
    lines.append("This is a plain text field built from a subset of the recovered columns.")
    lines.append("It is not the full recovery output. It is just a combined text version of selected recovered fields for search or embeddings.")
    lines.append("If you want the full recovery output, use `recovered_text_fields_json` instead.")
    lines.append("")

lines.append("## Column table")
lines.append("")
lines.append("| Column | Type | Coverage | Description |")
lines.append("|---|---|---:|---|")

for col in df.columns:
    non_null = df[col].notna().sum()
    pct = round((non_null / rows) * 100, 2)
    dtype = str(df[col].dtype)
    desc = DESCRIPTIONS.get(col, "Description not provided.")
    lines.append(f"| `{col}` | `{dtype}` | {pct}% | {desc} |")

lines.append("")
lines.append("## Original columns kept")
lines.append("")
for c in orig_cols:
    if c in df.columns:
        lines.append(f"- `{c}`")

lines.append("")
lines.append("## Recovered columns added")
lines.append("")
for c in recovered_cols:
    lines.append(f"- `{c}`")

with open(OUT, "w") as f:
    f.write("\n".join(lines))

print("README column documentation written to:")
print(OUT)