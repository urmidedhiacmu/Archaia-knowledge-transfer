# vocab.py
# Builds constrained value vocabulary per field from train split only.
# Outputs: data/vocab.json

import os, json
import pandas as pd

TRAIN_PATH = '/home/udedhia/archaia_project/archaia_impute/data/v4_train.parquet'
OUTPUT_DIR = '/home/udedhia/archaia_project/archaia_impute/data'

IMPUTE_FIELDS = [
    'recovered_material', 'recovered_object_type', 'recovered_condition',
    'recovered_period', 'recovered_description'
]
TOP_N_GLOBAL    = 50
TOP_N_PER_CLASS = 20
TOP_N_PER_PROJ  = 20

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading train split...')
    df = pd.read_parquet(TRAIN_PATH)
    print(f'  {len(df):,} artifacts')

    vocab = {}

    for field in IMPUTE_FIELDS:
        if field == 'recovered_description':
            continue
        print(f'\n{field}')
        vocab[field] = {}

        # global vocab
        vals = df[field].dropna().astype(str)
        vals = vals[~vals.isin(['nan','None',''])]
        global_vocab = vals.value_counts().head(TOP_N_GLOBAL).index.tolist()
        vocab[field]['_global'] = global_vocab
        print(f'  global: {len(global_vocab)} values')

        # per item_class vocab
        vocab[field]['_by_class'] = {}
        for cls, grp in df.groupby('item_class_label'):
            cls_vals = grp[field].dropna().astype(str)
            cls_vals = cls_vals[~cls_vals.isin(['nan','None',''])]
            if len(cls_vals) < 3: continue
            vocab[field]['_by_class'][cls] = (
                cls_vals.value_counts().head(TOP_N_PER_CLASS).index.tolist()
            )

        # per project vocab
        vocab[field]['_by_project'] = {}
        for proj, grp in df.groupby('project_label'):
            proj_vals = grp[field].dropna().astype(str)
            proj_vals = proj_vals[~proj_vals.isin(['nan','None',''])]
            if len(proj_vals) < 3: continue
            vocab[field]['_by_project'][proj] = (
                proj_vals.value_counts().head(TOP_N_PER_PROJ).index.tolist()
            )

    out_path = os.path.join(OUTPUT_DIR, 'vocab.json')
    with open(out_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f'\nWrote {out_path}')

if __name__ == '__main__':
    main()