# dedup_image_paths.py
# Adds deduplicated image_paths to v4 by collapsing same-URL-stem duplicates

import pandas as pd
import numpy as np
import os

V4      = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
TSV     = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'
OUT_V4  = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'  # overwrite

print("Loading TSV...")
tsv = pd.read_csv(TSV, sep='\t')
# row index -> url stem
idx_to_stem = {
    i: os.path.splitext(os.path.basename(str(url)))[0]
    for i, url in tsv['url'].items()
}
print(f"  {len(idx_to_stem):,} TSV rows indexed")

print("Loading v4...")
df = pd.read_parquet(V4)
print(f"  {len(df):,} rows")

def dedup_paths(paths):
    if not isinstance(paths, (list, np.ndarray)):
        return []
    seen = set()
    out  = []
    for p in paths:
        try:
            row_idx  = int(os.path.splitext(os.path.basename(str(p)))[0])
            url_stem = idx_to_stem.get(row_idx, str(row_idx))
        except:
            url_stem = str(p)
        if url_stem not in seen:
            seen.add(url_stem)
            out.append(p)
    return out

print("Deduplicating image_paths...")
df['image_paths']  = df['image_paths'].apply(
    lambda v: dedup_paths(list(v) if isinstance(v, np.ndarray) else v)
)
df['image_count_y'] = df['image_paths'].apply(len)

before = 102140  # from earlier output
after  = df['image_count_y'].sum()
print(f"  Total image refs before: {before:,}")
print(f"  Total image refs after:  {after:,}")
print(f"  Removed:                 {before - after:,}")
print(f"  Mean per artifact:       {df['image_count_y'].mean():.2f}")
print(f"  Max per artifact:        {df['image_count_y'].max()}")

print("Saving...")
df.to_parquet(OUT_V4, index=False)
print(f"Done. Saved to {OUT_V4}")