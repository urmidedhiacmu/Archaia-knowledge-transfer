# split.py
# Stratified 85/15 split of v4 by (project_label, item_class_label)
# Outputs:
#   data/v4_train.parquet  (85%)
#   data/v4_eval.parquet   (15%)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

V4_PATH    = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
OUTPUT_DIR = '/home/udedhia/archaia_project/archaia_impute/data'

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading v4...')
    df = pd.read_parquet(V4_PATH)
    print(f'  {len(df):,} total artifacts')

    # stratify by (project_label, item_class_label) combined
    df['_strat'] = df['project_label'].astype(str) + '||' + df['item_class_label'].astype(str)

    # groups with < 2 members can't be stratified — put them all in train
    counts = df['_strat'].value_counts()
    rare   = counts[counts < 2].index
    rare_df   = df[df['_strat'].isin(rare)].copy()
    common_df = df[~df['_strat'].isin(rare)].copy()
    print(f'  {len(rare_df):,} artifacts in rare strata (all go to train)')
    print(f'  {len(common_df):,} artifacts available for stratified split')

    train_common, eval_df = train_test_split(
        common_df,
        test_size=0.15,
        stratify=common_df['_strat'],
        random_state=42
    )

    train_df = pd.concat([train_common, rare_df]).drop(columns=['_strat'])
    eval_df  = eval_df.drop(columns=['_strat'])

    print(f'\nSplit result:')
    print(f'  train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)')
    print(f'  eval:  {len(eval_df):,} ({len(eval_df)/len(df)*100:.1f}%)')

    # verify no overlap
    train_uuids = set(train_df['uuid_hex'])
    eval_uuids  = set(eval_df['uuid_hex'])
    overlap = train_uuids & eval_uuids
    print(f'  overlap: {len(overlap)} (must be 0)')
    assert len(overlap) == 0, 'OVERLAP DETECTED'

    # class distribution comparison
    print(f'\nClass distribution:')
    print(f'  {"class":<30} {"train":>8} {"eval":>8}')
    for cls in sorted(df['item_class_label'].unique()):
        tc = (train_df['item_class_label'] == cls).sum()
        ec = (eval_df['item_class_label'] == cls).sum()
        print(f'  {cls:<30} {tc:>8,} {ec:>8,}')

    train_path = os.path.join(OUTPUT_DIR, 'v4_train.parquet')
    eval_path  = os.path.join(OUTPUT_DIR, 'v4_eval.parquet')
    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    print(f'\nWrote {train_path}')
    print(f'Wrote {eval_path}')

if __name__ == '__main__':
    main()