# audit.py
# Defines is_missing() — single source of truth used by all downstream scripts.
# Also produces data/audit_report.json summarizing missingness per field.
#
# Run:
#   cd /home/udedhia/archaia_project/archaia_impute
#   source ~/archaia_env/bin/activate
#   python3 00_data_prep/audit.py

import os, json
import pandas as pd

PARQUET_V4 = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
OUTPUT_DIR = '/home/udedhia/archaia_project/archaia_impute/data'

IMPUTE_FIELDS = {
    'high':   ['earliest', 'latest', 'latitude', 'longitude', 'recovered_period'],
    'medium': ['recovered_material', 'recovered_object_type', 'recovered_condition',
               'recovered_decorative_technique', 'recovered_chronotype'],
    'low':    ['recovered_description', 'recovered_fabric_group'],
}

ALL_IMPUTE_FIELDS = [f for tier in IMPUTE_FIELDS.values() for f in tier]

SEMANTICALLY_EMPTY = {'unknown', 'not recorded', 'n/a', 'na', '-', '?', 'none', 'nan', ''}

def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in SEMANTICALLY_EMPTY

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading v4 parquet...')
    df = pd.read_parquet(PARQUET_V4)
    print(f'  {len(df)} rows, {len(df.columns)} cols')

    report = {}
    print(f'\n{"field":<38} {"tier":<8} {"populated":>10} {"missing%":>9}')
    print('-' * 70)

    for tier, fields in IMPUTE_FIELDS.items():
        for col in fields:
            s = df[col]
            hard_null = int(s.isna().sum())
            soft_null = int(
                s.dropna().astype(str).str.strip().str.lower()
                .isin(SEMANTICALLY_EMPTY).sum()
            )
            populated   = int(len(s) - hard_null - soft_null)
            pct_missing = round((hard_null + soft_null) / len(s) * 100, 1)

            clean = s.dropna().astype(str)
            clean = clean[~clean.str.strip().str.lower().isin(SEMANTICALLY_EMPTY)]
            top5  = clean.value_counts().head(5).to_dict()

            report[col] = {
                'tier':        tier,
                'total':       len(s),
                'hard_null':   hard_null,
                'soft_null':   soft_null,
                'populated':   populated,
                'pct_missing': pct_missing,
                'top5':        top5,
            }
            print(f'  {col:<36} {tier:<8} {populated:>10,} {pct_missing:>8}%')

    out_path = os.path.join(OUTPUT_DIR, 'audit_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nWrote {out_path}')

if __name__ == '__main__':
    main()
