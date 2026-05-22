#!/usr/bin/env python3
import pandas as pd

DATA_DIR = '/home/udedhia/archaia_project/data'

print("Loading data...")
manifest = pd.read_parquet(f'{DATA_DIR}/oc_all_manifest.parquet')
resources = pd.read_parquet(f'{DATA_DIR}/oc_all_resources.parquet')

print("\n" + "="*80)
print("RESOURCES TABLE EXPLORATION")
print("="*80)

print("\nColumns in resources:")
print(resources.columns.tolist())

print("\nFirst 5 rows:")
print(resources.head())

print("\nURI examples (first 20):")
print(resources['uri'].head(20).tolist())

print("\nURI patterns - looking for images:")
image_like = resources[resources['uri'].str.contains('.jpg|.png|.jpeg|.gif', case=False, na=False)]
print(f"\nRows with image extensions: {len(image_like):,}")
print("\nSample image URIs:")
print(image_like['uri'].head(10).tolist())

print("\n" + "="*80)
print("MANIFEST TABLE - ITEM TYPES")
print("="*80)

print("\nItem types in manifest:")
print(manifest['item_type'].value_counts())

print("\nChecking if resources have item_uuid:")
print(f"Resources with item_uuid: {resources['item_uuid'].notna().sum():,}")

# Try the merge to see what happens
print("\n" + "="*80)
print("TESTING MERGE")
print("="*80)

resources_with_type = resources.merge(
    manifest[['uuid', 'item_type']],
    left_on='item_uuid',
    right_on='uuid',
    how='left'
)

print(f"\nAfter merge: {len(resources_with_type):,} rows")
print("\nItem types after merge:")
print(resources_with_type['item_type'].value_counts(dropna=False))

media = resources_with_type[resources_with_type['item_type'] == 'media']
print(f"\nRows where item_type == 'media': {len(media):,}")

if len(media) > 0:
    print("\nSample media URIs:")
    print(media['uri'].head(10).tolist())
