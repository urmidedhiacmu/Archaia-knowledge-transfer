# entropy.py
# Computes Shannon entropy per (project_label, item_class_label, field)
# from train split only. Low entropy = stat_imputable (use top value directly).
# Outputs: data/field_entropy.json

import os, json
import numpy as np
import pandas as pd

TRAIN_PATH = '/home/udedhia/archaia_project/archaia_impute/data/v4_train.parquet'
OUTPUT_DIR = '/home/udedhia/archaia_project/archaia_impute/data'

IMPUTE_FIELDS = [
    'recovered_material', 'recovered_object_type', 'recovered_condition',
    'recovered_period', 'recovered_description'
]
ENTROPY_THRESHOLD = 0.5  # below this → stat_imputable

def shannon_entropy(series):
    counts = series.value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts + 1e-10)))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading train split...')
    df = pd.read_parquet(TRAIN_PATH)
    print(f'  {len(df):,} artifacts')

    result = {}
    print(f'\n{"field":<35} {"project|class":<50} {"entropy":>8} {"top_val":<30} {"top_freq":>8} {"stat_imp"}')
    print('-' * 140)

    for field in IMPUTE_FIELDS:
        if field == 'recovered_description':
            continue  # skip free text
        result[field] = {}
        for (proj, cls), grp in df.groupby(['project_label', 'item_class_label']):
            vals = grp[field].dropna().astype(str)
            vals = vals[~vals.isin(['nan','None',''])]
            if len(vals) < 3:
                continue
            ent       = shannon_entropy(vals)
            top_val   = vals.value_counts().index[0]
            top_freq  = float(vals.value_counts(normalize=True).iloc[0])
            stat_imp  = ent < ENTROPY_THRESHOLD

            key = f'{proj}||{cls}'
            result[field][key] = {
                'entropy':       round(ent, 4),
                'top_value':     top_val,
                'top_freq':      round(top_freq, 4),
                'stat_imputable': stat_imp,
                'n':             len(vals),
            }
            if stat_imp:
                print(f'  {field:<33} {key[:48]:<50} {ent:>8.3f} {top_val[:28]:<30} {top_freq:>8.2%} YES')

    out_path = os.path.join(OUTPUT_DIR, 'field_entropy.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nWrote {out_path}')

    # summary
    for field in result:
        total    = len(result[field])
        stat_imp = sum(1 for v in result[field].values() if v['stat_imputable'])
        print(f'  {field:<35} {stat_imp:>4} / {total:>4} groups stat-imputable')

if __name__ == '__main__':
    main()