import pandas as pd

# Load dataset
DATA_PATH = "/data/group_data/dei-group/archaia/archaia_final_dataset.parquet"
df = pd.read_parquet(DATA_PATH)

# Useful columns list (your final schema)
cols = [
    # Artifact description
    "label", "slug", "project_label", "metadata",

    # Geographic
    "geo_note", "geometry_type", "geometry",
    "latitude", "longitude",
    "geo_depth", "geo_specificity_y", "geo_zoom",

    # Temporal
    "earliest", "start", "stop", "latest", "chrono_depth",

    # Classification
    "reference_type", "quality_score", "is_best",

    # External links
    "pleiades_id", "wikidata_id", "geonames_id",

    # Media
    "image_paths", "image_count_y",

    # Contextual provenance
    "project_uuid", "context_uuid", "item_class_uuid"
]

# Keep only columns that exist
cols = [c for c in cols if c in df.columns]

# Sample rows
sample_df = df[cols].sample(n=20, random_state=42)

# Save to CSV
sample_df.to_csv("archaia_useful_columns_sample.csv", index=False)

print("Saved: archaia_useful_columns_sample.csv")