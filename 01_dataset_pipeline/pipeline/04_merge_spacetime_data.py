import pandas as pd
import pyarrow.parquet as pq

ARTIFACT_PATH = "data/archaia_full_dataset_with_uuid_hex.csv"
PARQUET_PATH = "data/oc_all_manifest_cached_spacetime.parquet"

print("Loading artifacts...")
artifacts = pd.read_csv(ARTIFACT_PATH, low_memory=False)


pf = pq.ParquetFile(PARQUET_PATH)

results = []

for i in range(pf.num_row_groups):
    print(f"\nProcessing row group {i+1}/{pf.num_row_groups}")

    table = pf.read_row_group(i)
    sp = table.to_pandas()

    sp["item_uuid_hex"] = sp["item_uuid"].apply(
        lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x
    )

    merged = artifacts.merge(
        sp,
        left_on="uuid_hex",
        right_on="item_uuid_hex",
        how="left"
    )

    results.append(merged)

final = pd.concat(results, ignore_index=True)

print("Final shape:", final.shape)
print("Matched locations:", final["latitude"].notna().sum())
print("Matched time:", final["start"].notna().sum())

final.to_csv("data/artifacts_with_spacetime.csv", index=False)
final.to_parquet("data/artifacts_with_spacetime.parquet", index=False)
