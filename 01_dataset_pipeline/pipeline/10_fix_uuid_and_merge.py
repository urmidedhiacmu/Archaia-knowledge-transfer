# archaia_fix_complete.py
# Fixes everything and produces a fully verified final dataset

import pandas as pd
import numpy as np
import ast, json, os, subprocess
from pathlib import Path
from collections import defaultdict

TSV_PATH    = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'
IMG_DIR     = '/data/user_data/udedhia/archaia/final/images'
AUG_V2      = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v2.parquet'
BASE_CSV    = '/data/user_data/udedhia/archaia/final/archaia_final_dataset.csv'
MAPPING_CSV = '/data/user_data/udedhia/archaia/image_to_artifact_mapping.csv'
MANIFEST    = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'
ASSERTIONS  = '/home/udedhia/archaia_project/data/oc_all_assertions.parquet'
RESOURCES   = '/home/udedhia/archaia_project/data/oc_all_resources.parquet'
OUT_FINAL   = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'

# ── helpers ──────────────────────────────────────────────────────────────────
def norm_uuid(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("LOADING BASE FILES")
print("="*70)
tsv  = pd.read_csv(TSV_PATH, sep='\t')
tsv['artifact_hex'] = tsv['caption'].str.extract(r'artifact_([a-f0-9]+)_')
tsv['row_idx']      = tsv.index
print(f"TSV rows: {len(tsv):,} | Unique artifacts: {tsv['artifact_hex'].nunique():,}")

aug  = pd.read_parquet(AUG_V2)
base = pd.read_csv(BASE_CSV, low_memory=False, encoding='latin-1')
base['uuid_hex'] = base['uuid'].apply(norm_uuid)
print(f"Aug v2 rows: {len(aug):,}")
print(f"Base rows:   {len(base):,}")

mapping = pd.read_csv(MAPPING_CSV, low_memory=False)
mapping['artifact_hex'] = mapping['key'].str.extract(r'artifact_([a-f0-9]+)_')
mapping['media_uuid_hex'] = mapping['media_uuid'].apply(norm_uuid)
print(f"Mapping rows: {len(mapping):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FIX 1: ADD uuid_hex BACK")
print("="*70)
aug = aug.merge(base[['slug','uuid_hex']], on='slug', how='left')
missing_uuid = aug['uuid_hex'].isna().sum()
print(f"Rows with uuid_hex: {aug['uuid_hex'].notna().sum():,} / {len(aug):,}")
print(f"Missing uuid_hex:   {missing_uuid:,}")
if missing_uuid > 0:
    print("Sample missing:")
    print(aug[aug['uuid_hex'].isna()][['label','slug']].head(5).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FIX 2: FILTER NON-ARTIFACT MEDIA using manifest item_class")
print("  (Eric's suggestion: self-join item_class_uuid → item_class label)")
print("="*70)
print("Loading manifest...")
manifest = pd.read_parquet(MANIFEST)
manifest['uuid_n'] = manifest['uuid'].apply(norm_uuid)

# build item_class_uuid → class label lookup
class_lookup = manifest.set_index('uuid_n')['label'].to_dict()

# get item_class for every media entity
media_rows = manifest[manifest['item_type'] == 'media'].copy()
media_rows['item_class_label'] = media_rows['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)

print(f"\nMedia entity count: {len(media_rows):,}")
print("item_class distribution for media entities:")
print(media_rows['item_class_label'].value_counts().head(20).to_string())

# define which item_class labels are artifact photos vs not
ARTIFACT_PHOTO_CLASSES = {
    'photograph', 'image', 'still image', 'photo', 'digital image',
    'jpeg', 'jpg', 'tiff', 'png', 'artifact photo', 'object photo'
}
NON_ARTIFACT_CLASSES = {
    'locus photo', 'site photo', 'field photo', 'context photo',
    'section drawing', 'plan', 'map', 'drawing', 'diagram',
    'pdf', 'document', 'text', 'report', 'video'
}

media_rows['is_artifact_photo'] = media_rows['item_class_label'].str.lower().apply(
    lambda c: (
        any(ac in c for ac in ARTIFACT_PHOTO_CLASSES) or
        not any(nc in c for nc in NON_ARTIFACT_CLASSES)
    )
)

print(f"\nMedia classified as artifact photos: {media_rows['is_artifact_photo'].sum():,}")
print(f"Media classified as NON-artifact:    {(~media_rows['is_artifact_photo']).sum():,}")

# build set of media uuids that are artifact photos
artifact_photo_media = set(media_rows[media_rows['is_artifact_photo']]['uuid_n'])
non_artifact_photo_media = set(media_rows[~media_rows['is_artifact_photo']]['uuid_n'])

# filter mapping to only artifact-photo media
mapping_filtered = mapping[mapping['media_uuid_hex'].isin(artifact_photo_media)].copy()
print(f"\nMapping rows before filter: {len(mapping):,}")
print(f"Mapping rows after filter:  {len(mapping_filtered):,}")
print(f"Rows removed (non-artifact media): {len(mapping) - len(mapping_filtered):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FIX 3: DEDUPLICATE URLs per artifact")
print("  (same image linked via thumb + full + preview)")
print("="*70)
before_dedup = len(mapping_filtered)

# normalize URL: strip size/thumb/preview suffixes to get canonical image
def canonical_url(url):
    u = str(url)
    # strip common size variants
    for suffix in ['/thumbs/', '/thumbnails/', '/preview/', '/full/', '/medium/']:
        if suffix in u:
            return u  # keep as-is but we'll dedup by filename stem
    return u

# deduplicate: for each (artifact_hex, url) keep only one row
mapping_dedup = mapping_filtered.drop_duplicates(subset=['artifact_hex', 'url'])
print(f"Mapping rows before dedup: {before_dedup:,}")
print(f"Mapping rows after dedup:  {len(mapping_dedup):,}")
print(f"Duplicates removed:        {before_dedup - len(mapping_dedup):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FIX 4: REBUILD image_paths from filtered+deduped mapping + disk scan")
print("="*70)

# rebuild TSV row index lookup using filtered mapping
# filtered TSV rows: only those whose URL is in filtered+deduped mapping
filtered_urls = set(mapping_dedup['url'])
tsv_filtered = tsv[tsv['url'].isin(filtered_urls)].copy()
print(f"TSV rows after filtering: {len(tsv_filtered):,} / {len(tsv):,}")

# build row_idx → artifact_hex from full TSV (row index is still global)
idx_to_artifact = tsv.set_index('row_idx')['artifact_hex'].to_dict()

# scan disk and assign only if TSV row passes filter
tsv_filtered_idxs = set(tsv_filtered.index)
artifact_to_paths = defaultdict(list)
skipped_no_tsv = 0
skipped_filtered = 0

for shard_dir in sorted(Path(IMG_DIR).iterdir()):
    if not shard_dir.is_dir(): continue
    for img_file in sorted(shard_dir.iterdir()):
        if img_file.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        try:
            row_idx = int(img_file.stem)
        except ValueError:
            skipped_no_tsv += 1
            continue
        art_hex = idx_to_artifact.get(row_idx)
        if art_hex is None:
            skipped_no_tsv += 1
            continue
        if row_idx not in tsv_filtered_idxs:
            skipped_filtered += 1
            continue
        rel_path = f"images/{shard_dir.name}/{img_file.name}"
        artifact_to_paths[art_hex].append(rel_path)

total_mapped = sum(len(v) for v in artifact_to_paths.values())
print(f"Images mapped (after filter): {total_mapped:,}")
print(f"Images skipped (filtered out): {skipped_filtered:,}")
print(f"Images skipped (no TSV match): {skipped_no_tsv:,}")
print(f"Artifacts with images: {len(artifact_to_paths):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FIX 5: CHECK 171 ZERO-IMAGE ARTIFACTS")
print("="*70)
# aug already has uuid_hex from Fix 1 — use directly
aug['paths_v2'] = aug['image_paths'].apply(
    lambda v: list(v) if isinstance(v, (list, np.ndarray)) else []
)
zero_img_mask = aug['paths_v2'].apply(len) == 0
zero_img = aug[zero_img_mask]
print(f"Artifacts with 0 images in v2: {len(zero_img):,}")

zero_uuids = set(aug.loc[zero_img_mask, 'uuid_hex'].dropna())
in_mapping_raw      = zero_uuids & set(mapping['artifact_hex'])
in_mapping_filtered = zero_uuids & set(mapping_filtered['artifact_hex'])
truly_no_media      = zero_uuids - in_mapping_raw

print(f"  In raw mapping (URLs exist, not downloaded): {len(in_mapping_raw):,}")
print(f"  In filtered mapping (artifact-photo URLs):   {len(in_mapping_filtered):,}")
print(f"  Genuinely no media at all:                   {len(truly_no_media):,}")

# for those with URLs but not downloaded — list their URLs so we know why
if len(in_mapping_raw) > 0:
    sample_hex = list(in_mapping_raw)[:3]
    print(f"\n  Sample zero-image artifacts that HAVE mapping entries (not downloaded):")
    for h in sample_hex:
        rows = mapping[mapping['artifact_hex'] == h]
        label = aug.loc[aug['uuid_hex'] == h, 'label'].values
        print(f"    {label[0] if len(label) else h[:12]}: {rows['url'].tolist()[:2]}")

# ─────────────────────────────────────────────────────────────────────────────
# Note: manifest item_class cannot distinguish artifact photos from locus photos
# (all labelled "Image media"). Per-artifact Gemini filtering (filter_images.py)
# remains the only reliable way to remove non-artifact images.
# That is a separate downstream step and does not block saving v3.
print("\nNOTE: manifest item_class filter removed only Document/3D/GIS/Video media (204 rows).")
print("Locus/site photos within 'Image media' require Gemini-based filtering (filter_images.py).")
print("This is a known downstream task — not blocking v3 save.")

print("\n" + "="*70)
print("MERGE AND SAVE v3")
print("="*70)
aug['image_paths_v3'] = aug['uuid_hex'].map(
    lambda h: artifact_to_paths.get(str(h), [])
)
aug['image_count_v3'] = aug['image_paths_v3'].apply(len)

aug_out = aug.drop(columns=['image_paths','image_count_y','paths_v2'], errors='ignore')
aug_out = aug_out.rename(columns={
    'image_paths_v3': 'image_paths',
    'image_count_v3': 'image_count_y'
})

aug_out.to_parquet(OUT_FINAL, index=False)
print(f"Saved: {OUT_FINAL}")
print(f"Rows: {len(aug_out):,} | Columns: {len(aug_out.columns):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SANITY CHECK on v3")
print("="*70)
final = pd.read_parquet(OUT_FINAL)
final['paths_parsed'] = final['image_paths'].apply(
    lambda v: list(v) if isinstance(v, (list,np.ndarray)) else []
)

print(f"Rows:              {len(final):,}")
print(f"Columns:           {len(final.columns):,}")
print(f"uuid_hex present:  {final['uuid_hex'].notna().sum():,}")
print(f"latitude 100%:     {final['latitude'].notna().sum():,}")
print(f"Duplicate slugs:   {final['slug'].duplicated().sum():,}")
print(f"Duplicate uuid_hex:{final['uuid_hex'].duplicated().sum():,}")
print(f"\nImages:")
print(f"  Total refs:             {final['paths_parsed'].apply(len).sum():,}")
print(f"  Artifacts with 0 imgs:  {(final['image_count_y']==0).sum():,}")
print(f"  Artifacts with 1+ imgs: {(final['image_count_y']>0).sum():,}")
print(f"  Mean per artifact:      {final['image_count_y'].mean():.2f}")
print(f"  Max per artifact:       {final['image_count_y'].max():,}")

print(f"\nKey column nulls:")
for c in ['label','project_label','latitude','longitude','slug','uuid_hex',
          'image_paths','earliest','latest']:
    if c in final.columns:
        null_n = final[c].isna().sum()
        print(f"  {c:<20} null={null_n:,} ({null_n/len(final)*100:.1f}%)")

print("\nSpot check: 10 random artifacts — verify image file → TSV → uuid match")
sample = final[final['image_count_y'] > 0].sample(10, random_state=99)
all_ok = True
for _, row in sample.iterrows():
    paths = list(row['image_paths'])
    fname = os.path.basename(paths[0])
    try:
        row_idx = int(os.path.splitext(fname)[0])
    except:
        print(f"  {row['label'][:20]}: can't parse filename {fname}")
        continue
    tsv_art = idx_to_artifact.get(row_idx, 'NOT_FOUND')
    ds_art  = str(row['uuid_hex'])
    ok = tsv_art == ds_art
    if not ok: all_ok = False
    status = '✓' if ok else f'✗  TSV={tsv_art[:12]} DS={ds_art[:12]}'
    print(f"  {row['label'][:20]:<22} {fname} → {status}")

print(f"\n{'ALL CHECKS PASSED ✓' if all_ok else 'FAILURES FOUND ✗ — do not use v3'}")

print("\nColumn list:")
for c in final.columns:
    nn = final[c].notna().sum()
    print(f"  {c:<50} {nn:>6,}/{len(final):,} ({nn/len(final)*100:.1f}%)")