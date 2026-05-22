# archaia_check_item_class.py
# Check item_class distribution of the 31,624 subjects in v3
# to see how many are loci/structures vs actual artifacts

import pandas as pd
import ast

V3       = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v3.parquet'
MANIFEST = '/home/udedhia/archaia_project/data/oc_all_manifest.parquet'

df = pd.read_parquet(V3)
manifest = pd.read_parquet(MANIFEST)

def norm_uuid(v):
    if isinstance(v, bytes): return v.hex()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("b'") or s.startswith('b"'):
            try: return ast.literal_eval(s).hex()
            except: pass
        return s.lower().replace('-','')
    return str(v)

manifest['uuid_n'] = manifest['uuid'].apply(norm_uuid)
class_lookup = manifest.set_index('uuid_n')['label'].to_dict()

df['item_class_label'] = df['item_class_uuid'].apply(
    lambda v: class_lookup.get(norm_uuid(v), 'unknown')
)

print("="*70)
print("ITEM CLASS DISTRIBUTION OF 31,624 SUBJECTS IN V3")
print("="*70)
print(df['item_class_label'].value_counts().to_string())

print("\n\n" + "="*70)
print("SAMPLE OF EACH NON-ARTIFACT CLASS")
print("="*70)
for cls, grp in df.groupby('item_class_label'):
    print(f"\n{cls} ({len(grp):,} rows):")
    print(grp[['label','project_label']].head(5).to_string())