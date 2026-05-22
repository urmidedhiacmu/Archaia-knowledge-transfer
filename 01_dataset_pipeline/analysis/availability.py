import pandas as pd

df = pd.read_parquet("/data/group_data/dei-group/archaia/archaia_final_dataset.parquet")
total_rows = len(df)

categories = {
    "Artifact description": [
        "label", "slug", "project_label", "metadata"
    ],
    "Geographic": [
        "geo_note", "geometry_type", "geometry",
        "latitude", "longitude",
        "geo_depth", "geo_specificity_y", "geo_zoom"
    ],
    "Temporal": [
        "earliest", "start", "stop", "latest", "chrono_depth"
    ],
    "Classification / reference meaning": [
        "reference_type", "quality_score", "is_best"
    ],
    "External knowledge links": [
        "pleiades_id", "wikidata_id", "geonames_id"
    ],
    "Media info": [
        "image_paths", "image_count_y"
    ],
    "Contextual provenance": [
        "project_uuid", "context_uuid", "item_class_uuid"
    ]
}

# Function to compute availability
def availability(series):
    non_empty = (
        series.notna() &
        (series.astype(str).str.strip() != "")
    ).sum()
    percent = (non_empty / total_rows) * 100
    return int(non_empty), round(percent, 2)

rows = []

for category, cols in categories.items():
    for col in cols:
        if col in df.columns:
            count, pct = availability(df[col])
            rows.append({
                "Category": category,
                "Column": col,
                "Available Rows": count,
                "Percent Available (%)": pct
            })

final_df = pd.DataFrame(rows)

final_df = final_df.sort_values(["Category", "Column"])

final_df.to_csv("archaia_useful_columns_availability.csv", index=False)

print(final_df)