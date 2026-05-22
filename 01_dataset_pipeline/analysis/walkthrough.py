# archaia_pipeline_walkthrough.py
#
# Full pipeline walkthrough script for code review with Daphne
# Run section by section — each section is clearly labeled
#
# cd /home/udedhia/archaia_project/scripts
# source ~/archaia_env/bin/activate
# python3 -u archaia_pipeline_walkthrough.py

import pandas as pd
import numpy as np
import os, ast, json
from pathlib import Path
from collections import defaultdict

# ── PATHS ─────────────────────────────────────────────────────────────────
MANIFEST    = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'
ASSERTIONS  = '/home/udedhia/archaia_project/data/oc_all_assertions.parquet'
RESOURCES   = '/home/udedhia/archaia_project/data/oc_all_resources.parquet'
V4          = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
V1          = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet'
TSV         = '/data/user_data/udedhia/archaia/img2dataset_download_clean.tsv'
IMAGE_ROOT  = '/data/user_data/udedhia/archaia/final/images'

SEP = lambda title: print(f"\n{'='*70}\n  {title}\n{'='*70}")
sub = lambda title: print(f"\n  -- {title} --")

# ══════════════════════════════════════════════════════════════════════════
SEP("1. RAW DATA: THREE OPENCONTEXT PARQUETS")
# ══════════════════════════════════════════════════════════════════════════

sub("1.1 Manifest — entity registry")
manifest = pd.read_parquet(MANIFEST)
print(f"  Total entities:   {len(manifest):,}")
print(f"  item_type breakdown:")
print(manifest['item_type'].value_counts().head(8).to_string())

sub("1.2 Subjects (artifacts) in manifest")
subjects = manifest[manifest['item_type'] == 'subjects']
print(f"  Total subjects:   {len(subjects):,}")
print(f"  Sample labels:")
print(subjects['label'].head(5).to_string())

sub("1.3 Assertions — graph edges (36.9M rows)")
assertions_chunk = pd.read_parquet(ASSERTIONS, columns=[
    'subject_uuid','predicate_uuid','object_uuid',
    'obj_string','obj_double','obj_datetime'
])
print(f"  Total rows:       {len(assertions_chunk):,}")
print(f"  Columns:          {assertions_chunk.columns.tolist()}")
print(f"  Sample rows:")
print(assertions_chunk.head(3).to_string())

sub("1.4 Resources — media UUID to download URL")
resources = pd.read_parquet(RESOURCES)
print(f"  Total rows:       {len(resources):,}")
print(f"  Columns:          {resources.columns.tolist()}")
print(f"  Sample:")
print(resources[['item_uuid','uri']].head(3).to_string())


# ══════════════════════════════════════════════════════════════════════════
SEP("2. THE 3-HOP IMAGE JOIN")
# ══════════════════════════════════════════════════════════════════════════

print("""
  To get images for an artifact, you need three hops:

  artifact.uuid
    -> assertions[subject_uuid = artifact.uuid]  ->  object_uuid (= media UUID)
    -> manifest[uuid = media.uuid, item_type='media']
    -> resources[item_uuid = media.uuid]
    -> resources.uri  (download URL)

  A direct join of resources to artifacts returns NOTHING.
  resources.item_uuid points to media entities, not artifacts.
""")

# sub("2.1 Demonstrate: direct join returns nothing")
df_v4 = pd.read_parquet(V4)
sample_artifact = df_v4.iloc[0]
art_uuid = sample_artifact['uuid_hex']
# print(f"  Artifact: {sample_artifact['label']}  uuid_hex: {art_uuid}")

def norm(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

resources['item_uuid_norm'] = resources['item_uuid'].apply(norm)
direct = resources[resources['item_uuid_norm'] == art_uuid.lower().replace('-','')]
print(f"  Direct join result: {len(direct)} rows (expected 0)")

sub("2.1 Hop 1: artifact -> assertions -> media UUIDs")
import pyarrow.parquet as pq
print("  Reading assertions for this artifact (scanning, takes ~30s)...")
pf = pq.ParquetFile(ASSERTIONS)
hop1_rows = []
target = art_uuid.lower().replace('-','')
for i in range(pf.num_row_groups):
    batch = pf.read_row_group(i, columns=['subject_uuid','object_uuid','obj_string']).to_pandas()
    batch['subj_norm'] = batch['subject_uuid'].apply(norm)
    matched = batch[batch['subj_norm'] == target]
    if len(matched):
        hop1_rows.append(matched)
    if i % 5 == 0:
        print(f"    row group {i+1}/{pf.num_row_groups}...")

if hop1_rows:
    hop1 = pd.concat(hop1_rows)
    print(f"  Assertions for this artifact: {len(hop1)} rows")
    media_uuids = hop1['object_uuid'].apply(norm).dropna().unique().tolist()
    print(f"  Media UUIDs found: {media_uuids[:3]}")
else:
    print("  No assertions found for this artifact in chunk")
    media_uuids = []

sub("2.2 Hop 2+3: media UUID -> resources -> URL")
media_uuids = hop1['object_uuid'].apply(norm).unique().tolist()
media_mask = resources['item_uuid_norm'].isin(media_uuids)
final_urls = resources[media_mask]['uri']
print(f"  Image URLs found: {len(final_urls)}")
print(f"  Sample URLs:")
for url in final_urls.head(3):
    print(f"    {url}")


# ══════════════════════════════════════════════════════════════════════════
SEP("3. DATA FUNNEL")
# ══════════════════════════════════════════════════════════════════════════

funnel = [
    (2_165_383, "Total subjects in manifest"),
    (  207_062, "Media entities with at least 1 resource URL"),
    (  107_659, "Subjects with at least 1 downloadable image"),
    (   68_000, "After location + date filter (approx)"),
    (   33_412, "After image download and mapping reconstruction"),
    (   31_624, "After assertions recovery and augmentation (v1)"),
    (   22_607, "After removing non-artifact classes (v4, final)"),
]

print()
for count, label in funnel:
    bar = '█' * int(count / 100_000)
    print(f"  {count:>10,}  {label}")
    
print(f"\n  Biggest drop: location+date filter removes ~85% of image-bearing artifacts")
print(f"  These exist in OpenContext but have no coordinates or temporal data")


# ══════════════════════════════════════════════════════════════════════════
SEP("4. IMAGE MAPPING BUG AND FIX")
# ══════════════════════════════════════════════════════════════════════════

sub("4.1 How img2dataset names files")
print("""
  img2dataset input TSV format:
    url                                      caption
    https://archive.org/download/img1.jpg    artifact_abc123_001
    https://archive.org/download/img2.jpg    artifact_abc123_002
    https://archive.org/download/img3.jpg    artifact_def456_001

  Output files are named by TSV ROW INDEX, not by caption:
    row 0  ->  000000000.jpg
    row 1  ->  000000001.jpg
    row 2  ->  000000002.jpg

  The caption is only stored in a sidecar metadata JSON.
  The filename number is your only link back to the TSV.
""")

sub("4.2 What the original code did wrong")
print("""
  The original reconstruction assumed a fixed offset:
    artifact i -> images at rows [offset + i*n : offset + i*n + count]

  This breaks because artifacts have variable image counts (1 to 762).
  Row groupings in the TSV are not uniform.
  Result: every artifact got images from a different artifact — systematic offset.
""")

sub("4.3 The fix: read caption directly from TSV")
print("  Loading TSV...")
tsv = pd.read_csv(TSV, sep='\t', header=None, names=['url','caption'])
print(f"  TSV rows: {len(tsv):,}")
print(f"  Sample rows:")
print(tsv.head(3).to_string())

print("""
  Fix logic:
    1. Scan every image file on disk
    2. Parse filename number -> TSV row index
    3. Read caption at that row -> artifact hex
    4. Assign image path to that artifact
    5. Join back to dataset on uuid_hex
""")

sub("4.4 Spot check: verify the fix is correct")
SPOT_CHECK_ARTIFACTS = [
    {"label": "B2018.1.132",  "uuid": "b9cf166ff66b4d448cd1292071753161"},
    {"label": "Cat # A29477", "uuid": "090d74c0d96b4e854821 9609c7d62478"},
    {"label": "DT# 1031",     "uuid": "90696d5c90516e67bc00a55c4079def7"},
]

print(f"\n  {'Artifact':<20} {'UUID match':>12}  {'First image path'}")
print(f"  {'-'*60}")
for art in SPOT_CHECK_ARTIFACTS:
    uid = art['uuid'].replace(' ','').lower()
    row = df_v4[df_v4['uuid_hex'].str.lower().str.replace('-','') == uid]
    if len(row) == 0:
        print(f"  {art['label']:<20} NOT FOUND IN V4")
        continue
    paths = list(row.iloc[0]['image_paths'])
    if not paths:
        print(f"  {art['label']:<20} no images")
        continue
    first_path = paths[0]
    try:
        row_idx   = int(Path(first_path).stem)
        tsv_caption = tsv.iloc[row_idx]['caption']
        tsv_hex   = tsv_caption.split('_')[1] if '_' in tsv_caption else ''
        match     = '✓ MATCH' if tsv_hex == uid else f'✗ MISMATCH ({tsv_hex[:8]}...)'
    except Exception as e:
        match = f'error: {e}'
    print(f"  {art['label']:<20} {match:>12}  {first_path}")


# ══════════════════════════════════════════════════════════════════════════
SEP("5. ITEM_CLASS FILTERING (v1 -> v4)")
# ══════════════════════════════════════════════════════════════════════════

sub("5.1 Why non-artifact subjects were in the dataset")
print("""
  manifest item_type='subjects' includes:
    - Physical artifacts (what we want)
    - Loci (excavation spatial units)
    - Trenches, Survey Units, Sites
    - These all passed the location+date filter because loci have coordinates
""")

sub("5.2 item_class breakdown in v4 (after filtering)")
print(df_v4['item_class_label'].value_counts().to_string())

sub("5.3 What was removed")
removed = [
    ("Locus",       6947, "Spatial subdivision of excavation"),
    ("Survey Unit", 1699, "Geographic zone for surface survey"),
    ("Unit",         179, "Stratigraphic excavation unit"),
    ("Site",         134, "Entire site record"),
    ("Trench",        50, "Excavation trench"),
    ("Context",        8, "Stratigraphic context"),
]
print(f"\n  {'Class':<15} {'Count':>7}  Description")
print(f"  {'-'*55}")
for cls, count, desc in removed:
    print(f"  {cls:<15} {count:>7,}  {desc}")
print(f"\n  Total removed: 9,017 rows (28.5% of v1)")
print(f"  31,624 -> 22,607")


# ══════════════════════════════════════════════════════════════════════════
SEP("6. IMAGE DEDUPLICATION")
# ══════════════════════════════════════════════════════════════════════════

sub("6.1 The duplication problem")
print("""
  OpenContext stores the same physical image under multiple URLs:
    https://archive.org/download/opencontext-24-19.../24-19...jpg   (original)
    https://storage.googleapis.com/.../ia-previews/24-19...jpg      (GCS preview)
    https://storage.googleapis.com/.../ia-thumbnails/24-19...jpg    (GCS thumbnail)

  All three have different URLs but identical filename stems.
  All three appear as separate rows in the TSV -> 3 separate downloaded files.
  All 3 get assigned to the same artifact -> artifact appears to have 3x images.
""")

sub("6.2 Before deduplication")
print(f"  Total image references: 102,140")
print(f"  Mean per artifact:      4.52")

sub("6.3 Two dedup passes")
print("""
  Pass 1: deduplicate on (artifact_hex, url) pairs
    Removes exact URL duplicates
    Removed: 8,299 rows

  Pass 2: deduplicate on URL filename stem
    Collapses archive.org + GCS preview + GCS thumbnail triples
    Removed: 24,026 rows
""")

sub("6.4 After deduplication")
print(f"  Total image references: 78,114")
print(f"  Mean per artifact:      {df_v4['image_count_y'].mean():.2f}")
print(f"  Max per artifact:       {df_v4['image_count_y'].max()} (Struct. 15124, Giza)")

sub("6.5 Live check: PC 20000039 (had 3 duplicate images in browser)")
row = df_v4[df_v4['label'] == 'PC 20000039']
if len(row):
    paths = list(row.iloc[0]['image_paths'])
    print(f"  Image paths after dedup: {len(paths)}")
    for p in paths:
        idx = int(Path(p).stem)
        url = tsv.iloc[idx]['url']
        print(f"    {p}  ->  {url}")


# ══════════════════════════════════════════════════════════════════════════
SEP("7. FINAL DATASET: v4 STATS")
# ══════════════════════════════════════════════════════════════════════════

sub("7.1 Shape and coverage")
print(f"  Rows:                   {len(df_v4):,}")
print(f"  Columns:                {len(df_v4.columns)}")
print(f"  uuid_hex coverage:      {df_v4['uuid_hex'].notna().sum():,} / {len(df_v4):,} (100%)")
print(f"  lat/lon coverage:       {df_v4['latitude'].notna().sum():,} / {len(df_v4):,} (100%)")
print(f"  temporal coverage:      {df_v4['earliest'].notna().sum():,} / {len(df_v4):,} ({df_v4['earliest'].notna().mean()*100:.1f}%)")
print(f"  recovered_text_json:    {df_v4['recovered_text_fields_json'].notna().sum():,} / {len(df_v4):,} ({df_v4['recovered_text_fields_json'].notna().mean()*100:.1f}%)")
print(f"  Total image refs:       {df_v4['image_count_y'].sum():,}")
print(f"  Artifacts with images:  {(df_v4['image_count_y'] > 0).sum():,} ({(df_v4['image_count_y'] > 0).mean()*100:.1f}%)")
print(f"  Mean images/artifact:   {df_v4['image_count_y'].mean():.2f}")
print(f"  Duplicate uuid_hex:     {df_v4['uuid_hex'].duplicated().sum()}")
print(f"  Duplicate slugs:        {df_v4['slug'].duplicated().sum()}")

sub("7.2 Sample artifact — full record")
sample = df_v4[df_v4['label'] == 'DT# 1031'].iloc[0]
print(f"  Label:        {sample['label']}")
print(f"  Project:      {sample['project_label']}")
print(f"  Class:        {sample['item_class_label']}")
print(f"  UUID:         {sample['uuid_hex']}")
print(f"  Lat/Lon:      {sample['latitude']}, {sample['longitude']}")
print(f"  Dates:        {sample['earliest']} to {sample['latest']}")
print(f"  Images:       {sample['image_count_y']}")
print(f"  Material:     {sample.get('recovered_material', 'N/A')}")
print(f"  Description:  {str(sample.get('recovered_description', ''))[:120]}")

sub("7.3 recovered_* field coverage")
rcols = [c for c in df_v4.columns if c.startswith('recovered_') and c != 'recovered_text_fields_json']
coverage = df_v4[rcols].notna().mean().sort_values(ascending=False) * 100
for col, pct in coverage.items():
    bar = '█' * int(pct / 5)
    print(f"  {col:<40} {pct:5.1f}%  {bar}")

sub("7.4 Top projects")
print(df_v4['project_label'].value_counts().head(10).to_string())

print(f"\n{'='*70}")
print(f"  WALKTHROUGH COMPLETE")
print(f"  Browser: https://urmidedhiacmu.github.io/Archaia-viewer/")
print(f"  HF:      https://huggingface.co/datasets/archaia/dataset_sample_100_v4")
print(f"{'='*70}\n")