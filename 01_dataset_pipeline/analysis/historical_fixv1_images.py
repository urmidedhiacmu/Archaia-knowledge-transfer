# fix_v1_image_paths.py

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

TSV_PATH    = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'
V1_PARQUET  = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet'
IMAGE_DIR   = '/data/user_data/udedhia/archaia/final/images'
OUT_PARQUET = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1_fixed.parquet'

print("Loading TSV...")
tsv = pd.read_csv(TSV_PATH, sep='\t')
tsv['artifact_hex'] = tsv['caption'].str.extract(r'artifact_([a-f0-9]+)_')
idx_to_artifact = tsv['artifact_hex'].to_dict()
print(f"  {len(tsv):,} rows")

print("Scanning images on disk...")
artifact_to_paths = defaultdict(list)
for shard_dir in sorted(Path(IMAGE_DIR).iterdir()):
    if not shard_dir.is_dir(): continue
    for img_file in sorted(shard_dir.iterdir()):
        if img_file.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        try:
            row_idx = int(img_file.stem)
        except ValueError:
            continue
        art_hex = idx_to_artifact.get(row_idx)
        if art_hex is None: continue
        artifact_to_paths[art_hex].append(f"images/{shard_dir.name}/{img_file.name}")

print(f"  {sum(len(v) for v in artifact_to_paths.values()):,} images mapped")

print("Loading v1...")
v1 = pd.read_parquet(V1_PARQUET)
print(f"  {len(v1):,} rows")

import ast, json
def norm_uuid(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

# join uuid_hex if not present
if 'uuid_hex' not in v1.columns:
    print("Loading v4 for uuid_hex lookup...")
    v4 = pd.read_parquet('/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet')
    slug_to_uuid = v4.set_index('slug')['uuid_hex'].to_dict()
    v1['uuid_hex'] = v1['slug'].map(slug_to_uuid)
    print(f"  uuid_hex matched: {v1['uuid_hex'].notna().sum():,} / {len(v1):,}")

print("Rebuilding image_paths...")
v1['image_paths'] = v1['uuid_hex'].map(
    lambda h: np.array(artifact_to_paths.get(str(h), []))
)
v1['image_count_y'] = v1['image_paths'].apply(len)

# spot check
print("\nSpot check (5 artifacts of interest):")
for label in ['PC 19990012', 'PC 20090218', 'DT# 1031']:
    rows = v1[v1['label'] == label]
    if len(rows):
        paths = list(rows.iloc[0]['image_paths'])
        print(f"  {label}: {len(paths)} images — {paths[:2]}")

print("\nSaving...")
v1['image_paths'] = v1['image_paths'].apply(
    lambda v: list(v) if isinstance(v, np.ndarray) else []
)
v1.to_parquet(OUT_PARQUET, index=False)
print(f"Done -> {OUT_PARQUET}")