# archaia_final_cleanup.py

import pandas as pd
import numpy as np
from pathlib import Path
import os

V3          = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'
OUT_FINAL   = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'  # overwrite
TSV_PATH    = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'

df = pd.read_parquet(V3)

print("="*70)
print("ISSUE 1: DROP temp columns")
print("="*70)
temp_cols = ['paths_parsed', 'paths_v2']
to_drop = [c for c in temp_cols if c in df.columns]
df = df.drop(columns=to_drop)
print(f"Dropped: {to_drop}")
print(f"Columns now: {len(df.columns)}")

print("\n" + "="*70)
print("ISSUE 2: INVESTIGATE 762-image artifact")
print("="*70)
df['n_paths'] = df['image_paths'].apply(
    lambda v: len(list(v)) if isinstance(v, (list, np.ndarray)) else 0
)
top5 = df.nlargest(5, 'n_paths')[['label','project_label','n_paths','uuid_hex']]
print("Top 5 artifacts by image count:")
print(top5.to_string())

# for the max artifact, inspect the actual paths
max_row = df.loc[df['n_paths'].idxmax()]
paths = list(max_row['image_paths'])
print(f"\nArtifact: {max_row['label']} ({max_row['project_label']})")
print(f"Total paths: {len(paths)}")
print(f"First 10 paths: {paths[:10]}")
print(f"Last 5 paths:   {paths[-5:]}")

# check if these are sequential (would indicate correct mapping)
# vs random (would indicate something wrong)
nums = []
for p in paths:
    try:
        nums.append(int(os.path.splitext(os.path.basename(p))[0]))
    except:
        pass
if nums:
    nums_sorted = sorted(nums)
    print(f"\nFilename numbers: min={min(nums)}, max={max(nums)}, count={len(nums)}")
    print(f"Are they sequential? gaps={sum(1 for a,b in zip(nums_sorted, nums_sorted[1:]) if b-a>1)}")
    # check TSV to see what URLs these correspond to
    tsv = pd.read_csv(TSV_PATH, sep='\t')
    tsv_sample = tsv.iloc[nums[:5]][['url','caption']]
    print(f"\nTSV entries for first 5 paths of this artifact:")
    print(tsv_sample.to_string())

print("\n" + "="*70)
print("ISSUE 2b: DEDUP image_paths within each artifact")
print("  (remove same filename appearing multiple times)")
print("="*70)
before_total = df['n_paths'].sum()

def dedup_paths(paths):
    if not isinstance(paths, (list, np.ndarray)):
        return []
    seen = set()
    out = []
    for p in paths:
        fname = os.path.basename(str(p))
        if fname not in seen:
            seen.add(fname)
            out.append(p)
    return out

df['image_paths'] = df['image_paths'].apply(
    lambda v: dedup_paths(list(v) if isinstance(v, (list,np.ndarray)) else [])
)
df['image_count_y'] = df['image_paths'].apply(len)
df['n_paths'] = df['image_count_y']

after_total = df['n_paths'].sum()
print(f"Total paths before intra-artifact dedup: {before_total:,}")
print(f"Total paths after  intra-artifact dedup: {after_total:,}")
print(f"Removed: {before_total - after_total:,}")
print(f"\nNew max images per artifact: {df['n_paths'].max()}")
print("New top 5:")
print(df.nlargest(5,'n_paths')[['label','project_label','n_paths']].to_string())

print("\n" + "="*70)
print("FINAL SAVE AND SANITY CHECK")
print("="*70)
df = df.drop(columns=['n_paths'])
df.to_parquet(OUT_FINAL, index=False)
print(f"Saved: {OUT_FINAL}")

# reload and verify
final = pd.read_parquet(OUT_FINAL)
final_paths = final['image_paths'].apply(
    lambda v: len(list(v)) if isinstance(v, (list,np.ndarray)) else 0
)
print(f"\nRows:               {len(final):,}")
print(f"Columns:            {len(final.columns):,}")
print(f"Temp cols present:  {[c for c in ['paths_parsed','paths_v2','n_paths'] if c in final.columns]}")
print(f"uuid_hex 100%:      {final['uuid_hex'].notna().all()}")
print(f"No duplicate slugs: {not final['slug'].duplicated().any()}")
print(f"Total image refs:   {final_paths.sum():,}")
print(f"Artifacts w/ imgs:  {(final_paths > 0).sum():,}")
print(f"Artifacts w/o imgs: {(final_paths == 0).sum():,}")
print(f"Max imgs/artifact:  {final_paths.max()}")
print(f"Mean imgs/artifact: {final_paths.mean():.2f}")
print(f"\nAll columns:")
for c in final.columns:
    nn = final[c].notna().sum()
    print(f"  {c:<50} {nn:>6,}/{len(final):,} ({nn/len(final)*100:.1f}%)")