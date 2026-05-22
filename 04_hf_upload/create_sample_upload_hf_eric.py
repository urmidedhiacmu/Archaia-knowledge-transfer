# create_sample_upload_hf_eric.py — updated for v4

import os, json, shutil, ast
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, login

login(token="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")

PARQUET    = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'  
IMAGE_DIR  = '/data/user_data/udedhia/archaia/final'
SAMPLE_DIR = '/home/udedhia/archaia_project/scripts/archaia_sample_100_v4'
HF_REPO    = 'archaia/dataset_sample_100_v4'
SAMPLE_N   = 100

# Prefer these classes for the sample — actual portable objects
PREFERRED_CLASSES = {
    'Object', 'Pottery', 'Coin', 'Architectural Element',
    'Lithic', 'Animal Bone', 'Groundstone', 'Sculpture',
    'Glass', 'Shell', 'Human Bone', 'Biological record'
}

def get_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try:    return json.loads(raw)
        except: return ast.literal_eval(raw)
    return list(raw) if raw is not None else []

def main():
    print("Loading v4 parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"  Full dataset: {len(df)} rows, {len(df.columns)} columns")
    if 'item_class_label' in df.columns:
        print(f"  item_class breakdown:\n{df['item_class_label'].value_counts().to_string()}")

    rcols = [c for c in df.columns if c.startswith('recovered_') and c != 'recovered_text_fields_json']
    df['_richness'] = df[rcols].notna().sum(axis=1)

    # Prefer confirmed artifact classes; fall back to full dataset if needed
    if 'item_class_label' in df.columns:
        preferred = df[df['item_class_label'].isin(PREFERRED_CLASSES)]
        fallback  = df[~df['item_class_label'].isin(PREFERRED_CLASSES)]
        print(f"  Preferred class rows: {len(preferred)}, fallback: {len(fallback)}")
    else:
        preferred = df
        fallback  = df.iloc[0:0]

    top_projects = preferred['project_label'].value_counts().head(10).index.tolist()
    frames = []
    per_project = max(5, SAMPLE_N // len(top_projects))
    for proj in top_projects:
        sub = preferred[preferred['project_label'] == proj].sort_values('_richness', ascending=False)
        frames.append(sub.head(per_project))

    sample = pd.concat(frames)
    sample = sample[~sample.index.duplicated(keep='first')]

    # top up from fallback if short
    if len(sample) < SAMPLE_N:
        needed = SAMPLE_N - len(sample)
        extra  = fallback.sort_values('_richness', ascending=False).head(needed)
        sample = pd.concat([sample, extra])
        print(f"  Topped up with {len(extra)} fallback rows")

    sample = sample.head(SAMPLE_N).copy()
    sample = sample.drop(columns=['_richness'])
    print(f"  Sample: {len(sample)} artifacts from {sample['project_label'].nunique()} projects")
    print(f"  Projects: {sample['project_label'].value_counts().to_dict()}")
    if 'item_class_label' in sample.columns:
        print(f"  Classes: {sample['item_class_label'].value_counts().to_dict()}")

    # copy images
    sample_images_dir = os.path.join(SAMPLE_DIR, 'images')
    os.makedirs(sample_images_dir, exist_ok=True)

    new_image_paths = []
    total_images, missing_images = 0, 0

    for _, row in sample.iterrows():
        paths = get_paths(row)
        kept  = []
        for p in paths:
            src = os.path.join(IMAGE_DIR, p)
            if not os.path.exists(src):
                missing_images += 1
                continue
            dst = os.path.join(SAMPLE_DIR, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            kept.append(p)
            total_images += 1
        new_image_paths.append(json.dumps(kept))

    sample['image_paths'] = new_image_paths
    print(f"  Copied {total_images} images ({missing_images} missing/skipped)")

    parquet_out = os.path.join(SAMPLE_DIR, 'archaia_sample_100_v4.parquet')
    csv_out     = os.path.join(SAMPLE_DIR, 'archaia_sample_100_v4.csv')
    sample.to_parquet(parquet_out, index=False)
    sample.to_csv(csv_out, index=False)
    print(f"  Saved parquet and CSV")

    class_breakdown = ""
    if 'item_class_label' in sample.columns:
        class_breakdown = "\n## Artifact classes\n" + "\n".join(
            f"- {k}: {v}" for k, v in sample['item_class_label'].value_counts().items()
        )

    readme = f"""# Archaia Dataset V4 — 100 Artifact Sample

Sample of {len(sample)} confirmed artifacts from the Archaia v4 augmented dataset.
Sampled by metadata richness across top projects, prioritising confirmed artifact classes
(Object, Pottery, Coin, Lithic, etc.) over ambiguous classes (Feature, Structure).

## Source
Augmented from OpenContext data. Non-artifact subjects (Locus, Survey Unit, Trench, Site,
Unit, Context) have been removed in v4 via item_class filtering.

## Coverage
- Artifacts: {len(sample)}
- Projects: {sample['project_label'].nunique()}
- Columns: {len(sample.columns)}
- Images copied: {total_images}

## Projects included
{chr(10).join(f"- {k}: {v}" for k, v in sample['project_label'].value_counts().items())}
{class_breakdown}

## Key columns
- `label` — artifact catalog ID
- `project_label` — source excavation project
- `item_class_label` — artifact class (Object, Pottery, Coin, etc.)
- `image_paths` — JSON list of relative image paths
- `uuid_hex` — artifact UUID as hex string
- `recovered_description`, `recovered_note`, `recovered_material`, etc.
- `recovered_text_fields_json` — full JSON of all recovered assertion fields
- `latitude`, `longitude`, `earliest`, `latest`

## Image paths
Relative to dataset root, e.g. `images/00001/000012345.jpg`
"""
    with open(os.path.join(SAMPLE_DIR, 'README.md'), 'w') as f:
        f.write(readme)

    print(f"\nUploading to HuggingFace: {HF_REPO} ...")
    api = HfApi()
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(folder_path=SAMPLE_DIR, repo_id=HF_REPO, repo_type="dataset")
    print(f"\nDone! https://huggingface.co/datasets/{HF_REPO}")

if __name__ == '__main__':
    main()