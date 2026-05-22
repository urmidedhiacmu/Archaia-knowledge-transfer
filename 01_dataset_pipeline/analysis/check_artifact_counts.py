# verify_artifact_counts.py
# No assumptions — count everything directly from raw parquet

import pandas as pd
import ast
import numpy as np

MANIFEST   = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'
ASSERTIONS = '/home/udedhia/archaia_project/data/oc_all_assertions.parquet'
RESOURCES  = '/home/udedhia/archaia_project/data/oc_all_resources.parquet'
V3         = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'

def norm_uuid(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

print("Loading manifest...")
manifest = pd.read_parquet(MANIFEST)
manifest['uuid_n'] = manifest['uuid'].apply(norm_uuid)
class_lookup = manifest.set_index('uuid_n')['label'].to_dict()

print("="*70)
print("STEP 1: RAW COUNTS FROM MANIFEST")
print("="*70)
print(f"Total entities: {len(manifest):,}")
print(f"\nitem_type distribution:")
print(manifest['item_type'].value_counts().to_string())

subjects = manifest[manifest['item_type'] == 'subjects'].copy()
print(f"\nTotal subjects: {len(subjects):,}")

subjects['item_class_label'] = subjects['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)
print(f"\nitem_class distribution across ALL {len(subjects):,} subjects:")
print(subjects['item_class_label'].value_counts().to_string())

print("\n" + "="*70)
print("STEP 2: HOW MANY SUBJECTS HAVE AT LEAST 1 IMAGE?")
print("  (direct count from resources + assertions — no assumptions)")
print("="*70)
print("Loading resources...")
resources = pd.read_parquet(RESOURCES)
resources['item_n'] = resources['item_uuid'].apply(norm_uuid)

print("Loading assertions...")
assertions = pd.read_parquet(ASSERTIONS)
assertions['subj_n'] = assertions['subject_uuid'].apply(norm_uuid)
assertions['obj_n']  = assertions['object_uuid'].apply(norm_uuid)

# media uuids
media_uuids = set(manifest[manifest['item_type'] == 'media']['uuid_n'])
print(f"Total media entities: {len(media_uuids):,}")

# media uuids that have at least one resource (downloadable file)
media_with_resource = set(resources[resources['item_n'].isin(media_uuids)]['item_n'])
print(f"Media with at least 1 resource URL: {len(media_with_resource):,}")

# assertions linking subjects → media
print("Finding subject→media assertions...")
subj_to_media = assertions[assertions['obj_n'].isin(media_uuids)][['subj_n','obj_n']]
subj_to_media_with_resource = subj_to_media[subj_to_media['obj_n'].isin(media_with_resource)]

subjects_with_images = set(subj_to_media_with_resource['subj_n'])
print(f"\nSubjects with at least 1 downloadable image: {len(subjects_with_images):,}")

# break down by item_class
subjects['has_image'] = subjects['uuid_n'].isin(subjects_with_images)
print(f"\nBreakdown by item_class — subjects WITH images:")
print(subjects[subjects['has_image']]['item_class_label'].value_counts().to_string())

print("\n" + "="*70)
print("STEP 3: V3 ITEM CLASS BREAKDOWN vs RAW")
print("="*70)
v3 = pd.read_parquet(V3)
v3['item_class_label'] = v3['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)
print(f"V3 rows: {len(v3):,}")
print(f"\nV3 item_class breakdown:")
print(v3['item_class_label'].value_counts().to_string())

# what fraction of image-having subjects per class made it into v3?
print(f"\nCoverage: V3 rows vs all subjects-with-images per class:")
v3_counts   = v3['item_class_label'].value_counts().rename('in_v3')
raw_img_counts = subjects[subjects['has_image']]['item_class_label'].value_counts().rename('has_image_raw')
comparison = pd.concat([raw_img_counts, v3_counts], axis=1).fillna(0).astype(int)
comparison['pct_captured'] = (comparison['in_v3'] / comparison['has_image_raw'] * 100).round(1)
print(comparison.to_string())



print("\n\n" + "="*70)
print("\n\n" + "="*70)
print("\n\n" + "="*70)

print("PART 2")
print("="*70)

# archaia_classify_review.py
# Show real examples of each class so YOU can decide what's artifact vs not

import pandas as pd
import numpy as np
import ast

V3       = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'
MANIFEST = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'

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
df['n_imgs'] = df['image_paths'].apply(
    lambda v: len(list(v)) if isinstance(v, (list, np.ndarray)) else 0
)

print("="*70)
print("EVERY CLASS IN V3 WITH EXAMPLES")
print("Each row shows: label | project | n_images | recovered_description")
print("="*70)

for cls, grp in df.groupby('item_class_label', sort=False):
    grp_sorted = grp.sort_values('n_imgs', ascending=False)
    count = len(grp)
    total_imgs = grp['n_imgs'].sum()
    print(f"\n{'='*60}")
    print(f"CLASS: {cls}  ({count:,} rows, {total_imgs:,} total images)")
    print(f"{'='*60}")
    # show 5 examples with most images
    for _, row in grp_sorted.head(5).iterrows():
        desc = str(row.get('recovered_description',''))[:120] if pd.notna(row.get('recovered_description')) else ''
        note = str(row.get('recovered_note',''))[:80] if pd.notna(row.get('recovered_note')) else ''
        obj_type = str(row.get('recovered_object_type',''))[:60] if pd.notna(row.get('recovered_object_type')) else ''
        print(f"  label:       {row['label']}")
        print(f"  project:     {row['project_label']}")
        print(f"  n_images:    {row['n_imgs']}")
        if obj_type: print(f"  object_type: {obj_type}")
        if desc:     print(f"  description: {desc}")
        if note:     print(f"  note:        {note}")
        print()

print("\n" + "="*70)
print("SUMMARY TABLE — make your keep/remove decision per class")
print("="*70)
print(f"{'Class':<35} {'Rows':>6} {'Total imgs':>12} {'Mean imgs':>10}")
print("-"*70)
for cls, grp in df.groupby('item_class_label'):
    n = len(grp)
    total = grp['n_imgs'].sum()
    mean  = grp['n_imgs'].mean()
    print(f"  {cls:<33} {n:>6,} {total:>12,} {mean:>10.1f}")