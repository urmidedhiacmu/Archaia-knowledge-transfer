import pyarrow.parquet as pq
import pandas as pd

path = "data/oc_all_manifest_cached_spacetime.parquet"

pf = pq.ParquetFile(path)
table = pf.read_row_group(0)

sp = table.to_pandas()

# convert uuid bytes → hex string
sp["item_uuid_hex"] = sp["item_uuid"].apply(
    lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x
)

print(sp[["item_uuid", "item_uuid_hex"]].head())
