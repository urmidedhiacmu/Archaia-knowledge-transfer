# archaia_v4_final.py

import pandas as pd
import numpy as np
import ast

V3       = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'
MANIFEST = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'
OUT_V4   = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'

def norm_uuid(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

print("Loading...")
df       = pd.read_parquet(V3)
manifest = pd.read_parquet(MANIFEST)
manifest['uuid_n'] = manifest['uuid'].apply(norm_uuid)
class_lookup = manifest.set_index('uuid_n')['label'].to_dict()

df['item_class_label'] = df['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)

REMOVE = {'Locus', 'Survey Unit', 'Unit', 'Site', 'Trench', 'Context'}
KEEP   = {
    'Object', 'Pottery', 'Architectural Element', 'Feature', 'Sample',
    'Animal Bone', 'Coin', 'Lithic', 'Human Bone', 'Structure',
    'Groundstone', 'Sculpture', 'Biological record', 'Glass', 'Shell'
}

print("="*70)
print("FILTER")
print("="*70)
mask_keep   = ~df['item_class_label'].isin(REMOVE)
df_filtered = df[mask_keep].copy()

print(f"Before: {len(df):,}")
print(f"After:  {len(df_filtered):,}  (expected 22,607)")
print(f"Removed: {(~mask_keep).sum():,}  (expected 9,017)")
assert len(df_filtered) == 22607, f"Row count wrong: {len(df_filtered)}"

print("\nRemoved classes:")
for cls, n in df[~mask_keep]['item_class_label'].value_counts().items():
    print(f"  {cls:<20} {n:,}")

print("\nKept classes:")
for cls, n in df_filtered['item_class_label'].value_counts().items():
    print(f"  {cls:<30} {n:,}")

print("\n" + "="*70)
print("IMAGE STATS")
print("="*70)
df_filtered['n_imgs'] = df_filtered['image_paths'].apply(
    lambda v: len(list(v)) if isinstance(v, (list, np.ndarray)) else 0
)
print(f"Total image refs:    {df_filtered['n_imgs'].sum():,}")
print(f"Artifacts w/ images: {(df_filtered['n_imgs'] > 0).sum():,}")
print(f"Artifacts w/o imgs:  {(df_filtered['n_imgs'] == 0).sum():,}")
print(f"Max imgs/artifact:   {df_filtered['n_imgs'].max()}")
print(f"Mean imgs/artifact:  {df_filtered['n_imgs'].mean():.2f}")
print(f"\nTop 10 by image count:")
print(df_filtered.nlargest(10,'n_imgs')[['label','item_class_label','project_label','n_imgs']].to_string())

print("\n" + "="*70)
print("SAVE V4")
print("="*70)
df_out = df_filtered.drop(columns=['n_imgs'], errors='ignore')
df_out.to_parquet(OUT_V4, index=False)
print(f"Saved: {OUT_V4}")

print("\n" + "="*70)
print("FINAL SANITY CHECK — reload from disk")
print("="*70)
final = pd.read_parquet(OUT_V4)
final['item_class_label'] = final['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)
final_n = final['image_paths'].apply(
    lambda v: len(list(v)) if isinstance(v, (list, np.ndarray)) else 0
)

print(f"Rows:                {len(final):,}  ✓" if len(final)==22607 else f"Rows: {len(final):,}  ✗ WRONG")
print(f"Columns:             {len(final.columns):,}")
print(f"Temp cols:           {[c for c in ['paths_parsed','paths_v2','n_imgs'] if c in final.columns]}")
print(f"uuid_hex 100%:       {final['uuid_hex'].notna().all()}")
print(f"No dup slugs:        {not final['slug'].duplicated().any()}")
print(f"No dup uuid_hex:     {not final['uuid_hex'].duplicated().any()}")
print(f"latitude 100%:       {final['latitude'].notna().all()}")
print(f"No REMOVE classes:   {not final['item_class_label'].isin(REMOVE).any()}")
print(f"Total image refs:    {final_n.sum():,}")
print(f"Artifacts w/ images: {(final_n > 0).sum():,}")
print(f"Artifacts w/o imgs:  {(final_n == 0).sum():,}")
print(f"Max imgs/artifact:   {final_n.max()}")
print(f"Mean imgs/artifact:  {final_n.mean():.2f}")

print(f"\nKey nulls:")
for c in ['label','project_label','latitude','longitude','slug','uuid_hex','earliest','latest']:
    n = final[c].isna().sum()
    print(f"  {c:<20} null={n:,} ({n/len(final)*100:.1f}%)")

print(f"\nitem_class breakdown:")
print(final['item_class_label'].value_counts().to_string())

print(f"\nproject breakdown (top 10):")
print(final['project_label'].value_counts().head(10).to_string())

print(f"\nAll columns:")
for c in final.columns:
    nn = final[c].notna().sum()
    print(f"  {c:<50} {nn:>6,}/{len(final):,} ({nn/len(final)*100:.1f}%)")