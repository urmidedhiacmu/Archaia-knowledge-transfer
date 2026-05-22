# rebuild_image_paths.py
# Rebuilds correct image_paths for every artifact from scratch
# using: disk filename → TSV row index → caption → artifact_hex

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

TSV_PATH    = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'
IMG_DIR     = '/data/user_data/udedhia/archaia/final/images'
AUG_PARQUET = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet'
BASE_CSV    = '/data/user_data/udedhia/archaia/final/archaia_final_dataset.csv'
OUT_PARQUET = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v2.parquet'

print("Loading TSV...")
tsv = pd.read_csv(TSV_PATH, sep='\t')
tsv['artifact_hex'] = tsv['caption'].str.extract(r'artifact_([a-f0-9]+)_')
tsv['seq_num']      = tsv['caption'].str.extract(r'artifact_[a-f0-9]+_(\d+)').astype(float)
print(f"TSV rows: {len(tsv):,}")
print(f"Unique artifacts in TSV: {tsv['artifact_hex'].nunique():,}")

print("\nBuilding row_index → artifact_hex lookup...")
# TSV row index IS the filename number img2dataset uses
# row 0 → 000000000.jpg, row 14169 → 000014169.jpg
idx_to_artifact = tsv['artifact_hex'].to_dict()  # {int_index: hex_string}

print("\nScanning images on disk...")
artifact_to_paths = defaultdict(list)
skipped = 0
for shard_dir in sorted(Path(IMG_DIR).iterdir()):
    if not shard_dir.is_dir():
        continue
    for img_file in sorted(shard_dir.iterdir()):
        if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        try:
            row_idx = int(img_file.stem)  # e.g. 000014169 → 14169
        except ValueError:
            skipped += 1
            continue
        art_hex = idx_to_artifact.get(row_idx)
        if art_hex is None:
            skipped += 1
            continue
        # store as relative path
        rel_path = f"images/{shard_dir.name}/{img_file.name}"
        artifact_to_paths[art_hex].append(rel_path)

print(f"Images mapped: {sum(len(v) for v in artifact_to_paths.values()):,}")
print(f"Artifacts with images: {len(artifact_to_paths):,}")
print(f"Skipped (no TSV match): {skipped:,}")

# spot check
print("\nSpot check — first 3 artifacts:")
for hex_id, paths in list(artifact_to_paths.items())[:3]:
    tsv_rows = tsv[tsv['artifact_hex'] == hex_id]
    print(f"\n  artifact: {hex_id[:12]}...")
    print(f"  images: {paths[:2]}")
    print(f"  TSV URLs: {tsv_rows['url'].tolist()[:2]}")

print("\n\nLoading augmented parquet and merging corrected image_paths...")
aug = pd.read_parquet(AUG_PARQUET)
base = pd.read_csv(BASE_CSV, low_memory=False, encoding='latin-1')

# build uuid_hex → corrected paths
aug_merged = aug.merge(base[['slug','uuid_hex']], on='slug', how='left')
aug_merged['image_paths_correct'] = aug_merged['uuid_hex'].map(
    lambda h: np.array(artifact_to_paths.get(h, []))
)
aug_merged['image_count_correct'] = aug_merged['image_paths_correct'].apply(len)

print(f"\nBefore fix — artifacts with 0 images: {(aug['image_paths'].apply(lambda v: len(list(v)) if isinstance(v, (list,np.ndarray)) else 0) == 0).sum():,}")
print(f"After fix  — artifacts with 0 images: {(aug_merged['image_count_correct'] == 0).sum():,}")
print(f"After fix  — total image references:  {aug_merged['image_count_correct'].sum():,}")

# verify spot check matches TSV
print("\nVerification spot check (5 artifacts):")
sample = aug_merged[aug_merged['image_count_correct'] > 0].head(5)
all_ok = True
for _, row in sample.iterrows():
    hex_id = row['uuid_hex']
    new_paths = list(row['image_paths_correct'])
    tsv_rows = tsv[tsv['artifact_hex'] == hex_id]
    # check: filename of first new path → TSV row → same artifact?
    first = new_paths[0]  # images/00001/000014169.jpg
    fname = os.path.basename(first)
    row_idx = int(os.path.splitext(fname)[0])
    tsv_artifact = idx_to_artifact.get(row_idx, 'NOT FOUND')
    ok = tsv_artifact == hex_id
    if not ok: all_ok = False
    print(f"  {row.get('label','?')[:20]:<20} | {fname} → TSV artifact={tsv_artifact[:12]}... | ds artifact={hex_id[:12]}... | {'✓' if ok else '✗ MISMATCH'}")

print(f"\n{'All consistent ✓' if all_ok else 'MISMATCHES REMAIN ✗'}")

# replace just the save block at the bottom

if all_ok:
    print("\nSaving corrected dataset...")
    aug_merged = aug_merged.drop(columns=['image_paths', 'image_count_y', 'uuid_hex'], errors='ignore')
    aug_merged = aug_merged.rename(columns={
        'image_paths_correct': 'image_paths',
        'image_count_correct': 'image_count_y'
    })
    # convert numpy arrays → python lists so pyarrow can serialize
    aug_merged['image_paths'] = aug_merged['image_paths'].apply(
        lambda v: list(v) if isinstance(v, np.ndarray) else (v if isinstance(v, list) else [])
    )
    aug_merged.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved to: {OUT_PARQUET}")
    print(f"Rows: {len(aug_merged):,}")
    print(f"Columns: {aug_merged.columns.tolist()}")
else:
    print("\nNot saving — fix verification failed, investigate mismatches above.")