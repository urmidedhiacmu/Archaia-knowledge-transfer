# build_input.py
# Builds imputation_input.parquet from the 3 raw OC parquets.
# Artifacts with images, valid item_class, NOT in v4.
# Stopped before the spacetime filter — these are the imputation subjects.
#
# Run via sbatch:
#   sbatch archaia_impute/scripts/00_build_input.sh

import os
import pandas as pd
import uuid

DATA_DIR    = '/home/udedhia/archaia_project/data'
V4_PATH     = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
OUTPUT_DIR  = '/home/udedhia/archaia_project/archaia_impute/data'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'imputation_input.parquet')

MANIFEST_PATH   = os.path.join(DATA_DIR, 'oc_all_manifest.parquet')
ASSERTIONS_PATH = os.path.join(DATA_DIR, 'oc_all_assertions.parquet')
RESOURCES_PATH  = os.path.join(DATA_DIR, 'oc_all_resources.parquet')
SPACETIME_PATH  = os.path.join(DATA_DIR, 'oc_all_manifest_cached_spacetime.parquet')

KEEP_CLASS_UUIDS = {
    '000000006e24f30b83810e608046f9b5': 'Animal Bone',
    '000000006e249f217326d09207d65387': 'Architectural Element',
    '000000006e247c9fb487e9973c6da06d': 'Biological record',
    '000000006e2489991c4e817d853f967a': 'Coin',
    '000000006e2403af59315df0932d6be2': 'Feature',
    '000000006e24d0cddf8d4dbb6c2bd1dd': 'Glass',
    '000000006e24cdeddb78ae2e72ed3a89': 'Groundstone',
    '000000006e24885e0dd2c30a4228b7d3': 'Human Bone',
    '000000006e24158f2f6b36a9f8b58b32': 'Lithic',
    '000000006e2423399f63582218c3f76a': 'Object',
    '000000006e2465d0e0e427fa67c62929': 'Pottery',
    '000000006e240b7002300cc0398d6184': 'Sample',
    '000000006e24959457e4744439eb8fac': 'Sculpture',
    '000000006e24611fac23959560c71bc5': 'Shell',
    '000000006e24c7f5a7462ec91b5f1825': 'Structure',
}

# predicate_hex → (output_col, value_source)
# value_source: 'obj_uuid' = resolve object_uuid to entity label
#               'obj_string' = use obj_string directly
PREDICATE_MAP = {
    'de0970679ad05d48fb02e1905c46fefa': ('recovered_material',            'obj_uuid'),
    '7db79382743242a4fbc5ef760691905a': ('recovered_object_type',         'obj_uuid'),
    '4909306f310247a266a3561c296147bb': ('recovered_condition',           'obj_string'),
    '0b643ab938a44f450e41415c45cb7702': ('recovered_period',              'obj_uuid'),
    '13d9229565ea47f7ebf256c7667c6e5f': ('recovered_chronotype',          'obj_uuid'),
    'f07c30bc6c714c977893d61ff6d0b59b': ('recovered_decorative_technique','obj_uuid'),
    '423ba1ec3cd44dba40eb9474c1ae0d3a': ('recovered_fabric_group',        'obj_uuid'),
    '7dbb5cb7599f42d561ee1955cf898990': ('recovered_description',         'obj_string'),
    '5fa8fc7574f8725a489db727c033d79c': ('recovered_note',                'obj_string'),
}

MAX_IMAGES = 3

def bytes_to_hex(b):
    try:
        if isinstance(b, bytes): return uuid.UUID(bytes=b).hex
        return None
    except: return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load manifest, filter to valid artifact classes ────────────────
    print('Loading manifest...')
    manifest = pd.read_parquet(MANIFEST_PATH)
    print(f'  {len(manifest):,} total entities')

    manifest['uuid_hex']       = manifest['uuid'].apply(bytes_to_hex)
    manifest['item_class_hex'] = manifest['item_class_uuid'].apply(bytes_to_hex)
    manifest['context_hex']    = manifest['context_uuid'].apply(bytes_to_hex)
    manifest['project_hex']    = manifest['project_uuid'].apply(bytes_to_hex)

    # entity label lookup — used later for obj_uuid resolution
    entity_labels = dict(zip(manifest['uuid_hex'], manifest['label']))

    subjects = manifest[manifest['item_type'] == 'subjects'].copy()
    subjects = subjects[subjects['item_class_hex'].isin(KEEP_CLASS_UUIDS)].copy()
    subjects['item_class_label'] = subjects['item_class_hex'].map(KEEP_CLASS_UUIDS)
    print(f'  {len(subjects):,} after item_class filter')

    # ── 2. Remove v4 artifacts ────────────────────────────────────────────
    print('Removing v4 artifacts...')
    v4 = pd.read_parquet(V4_PATH)
    v4_uuids = set(v4['uuid_hex'].str.replace('-','').str.lower())
    subjects = subjects[~subjects['uuid_hex'].isin(v4_uuids)].copy()
    print(f'  {len(subjects):,} imputation candidates')
    subject_uuids = set(subjects['uuid_hex'].dropna())

    # ── 3. 3-hop image join ───────────────────────────────────────────────
    print('Loading resources...')
    resources = pd.read_parquet(RESOURCES_PATH)
    resources['item_hex'] = resources['item_uuid'].apply(bytes_to_hex)
    img_resources = resources[
        resources['uri'].str.contains(r'\.(jpg|jpeg|png)', case=False, na=False, regex=True)
    ][['item_hex','uri','rank']].copy()
    print(f'  {len(img_resources):,} image resource rows')
    media_with_images = set(img_resources['item_hex'].dropna())

    print('Loading assertions...')
    assertions = pd.read_parquet(ASSERTIONS_PATH)
    assertions['subj_hex'] = assertions['subject_uuid'].apply(bytes_to_hex)
    assertions['pred_hex'] = assertions['predicate_uuid'].apply(bytes_to_hex)
    assertions['obj_hex']  = assertions['object_uuid'].apply(bytes_to_hex)
    print(f'  {len(assertions):,} assertion rows')

    # filter assertions to our subjects only — do this once, reuse below
    subj_assertions = assertions[assertions['subj_hex'].isin(subject_uuids)].copy()
    print(f'  {len(subj_assertions):,} assertions for our subjects')

    # artifact → media links
    media_assertions = subj_assertions[
        subj_assertions['obj_hex'].isin(media_with_images)
    ][['subj_hex','obj_hex']].copy()
    print(f'  {len(media_assertions):,} artifact->media links')

    media_urls = media_assertions.merge(
        img_resources, left_on='obj_hex', right_on='item_hex', how='inner'
    )
    media_urls['url_stem'] = media_urls['uri'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    media_urls = media_urls.drop_duplicates(subset=['subj_hex','url_stem'])
    media_urls = media_urls.sort_values(['subj_hex','rank'])
    top_urls = (
        media_urls.groupby('subj_hex')['uri']
        .apply(lambda x: list(x.head(MAX_IMAGES)))
        .reset_index()
        .rename(columns={'uri':'image_urls','subj_hex':'uuid_hex'})
    )
    print(f'  {len(top_urls):,} artifacts with image URLs')

    subjects = subjects.merge(top_urls, on='uuid_hex', how='inner')
    print(f'  {len(subjects):,} subjects after image join')
    subject_uuids = set(subjects['uuid_hex'].dropna())

    # ── 4. Join spacetime ─────────────────────────────────────────────────
    print('Loading spacetime...')
    spacetime = pd.read_parquet(SPACETIME_PATH)
    spacetime['uuid_hex'] = spacetime['item_uuid'].apply(bytes_to_hex)
    spacetime_cols = [c for c in [
        'uuid_hex','latitude','longitude','earliest','latest',
        'start','stop','chrono_depth','geo_depth',
        'geometry','geometry_type','reference_type','quality_score'
    ] if c in spacetime.columns]
    spacetime = spacetime[spacetime_cols].drop_duplicates(subset='uuid_hex')
    subjects = subjects.merge(spacetime, on='uuid_hex', how='left')
    print(f'  have latitude: {subjects["latitude"].notna().sum():,}')
    print(f'  have earliest: {subjects["earliest"].notna().sum():,}')

    # ── 5. Extract recovered fields ───────────────────────────────────────
    print('Extracting recovered fields...')

    # filter subj_assertions to only our image-having subjects
    subj_assertions = subj_assertions[subj_assertions['subj_hex'].isin(subject_uuids)].copy()
    subj_assertions['obj_resolved'] = subj_assertions['obj_hex'].map(entity_labels)

    for pred_hex, (col_name, val_src) in PREDICATE_MAP.items():
        rows = subj_assertions[subj_assertions['pred_hex'] == pred_hex].copy()
        if val_src == 'obj_uuid':
            rows['value'] = rows['obj_resolved']
        else:
            rows['value'] = rows['obj_string']
        rows = rows[rows['value'].notna()].drop_duplicates(subset='subj_hex')
        rows = rows[['subj_hex','value']].rename(
            columns={'subj_hex':'uuid_hex','value':col_name}
        )
        subjects = subjects.merge(rows, on='uuid_hex', how='left')
        populated = subjects[col_name].notna().sum()
        print(f'  {col_name:<38} populated={populated:,}')

    # ── 6. Write ──────────────────────────────────────────────────────────
    keep_cols = [
        'uuid_hex','label','slug','project_hex','context_hex',
        'item_class_label','image_urls',
        'latitude','longitude','earliest','latest','start','stop',
        'chrono_depth','geo_depth','geometry','geometry_type',
        'reference_type','quality_score',
    ] + [col for _, (col, _) in PREDICATE_MAP.items()]
    keep_cols = [c for c in keep_cols if c in subjects.columns]
    subjects = subjects[keep_cols]

    print(f'\nFinal imputation_input.parquet:')
    print(f'  rows:    {len(subjects):,}')
    print(f'  columns: {len(subjects.columns)}')
    print(f'\nMissingness:')
    for col in keep_cols:
        if col in ['uuid_hex','label','slug','image_urls']: continue
        miss = subjects[col].isna().sum()
        pct  = miss / len(subjects) * 100
        print(f'  {col:<38} missing={miss:>6,} ({pct:.1f}%)')

    subjects.to_parquet(OUTPUT_PATH, index=False)
    print(f'\nWrote {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
