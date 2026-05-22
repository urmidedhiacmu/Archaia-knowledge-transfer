# build_combined_gh.py
#
# cd /home/udedhia/archaia_project/scripts
# source ~/archaia_env/bin/activate
# python3 -u build_combined_gh.py

import os, json, base64, ast
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image

ARTIFACT_SAMPLE_JSON = '/home/udedhia/archaia_project/gh_deploy/artifact_sample_100.json'
PARQUET_V4   = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
PARQUET_V1 = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1_fixed.parquet'
IMAGE_ROOT   = '/data/user_data/udedhia/archaia/final'
ABLATION_JSON = '/home/udedhia/archaia_project/uses_outputs/ablation_results.json'
OUTPUT_JSON  = '/home/udedhia/archaia_project/gh_deploy/combined_data.json'
OUTPUT_DIR   = '/home/udedhia/archaia_project/gh_deploy'

MAX_IMAGES   = 999
IMG_MAX_SIZE = 1000  
IMG_QUALITY  = 85   

KEY_FIELDS = [
    'label', 'project_label', 'item_class_label',
    'recovered_artifact_name', 'recovered_material', 'recovered_object_type',
    'recovered_period', 'recovered_condition', 'recovered_size',
    'recovered_description', 'recovered_note', 'recovered_function',
    'earliest', 'latest',
]

EXTRA_FIELDS = [
    'recovered_material_note', 'recovered_object_type_note',
    'recovered_chronotype', 'recovered_fabric_description',
    'recovered_fabric_group', 'recovered_munsell_color',
    'recovered_munsell_number', 'recovered_decorative_technique',
    'recovered_description_remarks', 'recovered_specific_context',
    'recovered_specific_location', 'recovered_location',
    'recovered_locus', 'recovered_locus_id',
    'recovered_registration_date', 'recovered_disposition',
    'recovered_text_fields_json',
    'latitude', 'longitude',
    'start', 'stop', 'chrono_depth',
    'slug', 'uuid_hex', 'context_uuid', 'item_class_uuid', 'project_uuid',
    'reference_type', 'is_best', 'quality_score',
    'geo_depth', 'geo_specificity_y', 'geo_zoom',
    'geometry', 'geometry_type',
    'image_count_y', 'metadata',
]

def get_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try:    return json.loads(raw)
        except: return ast.literal_eval(raw)
    if hasattr(raw, 'tolist'):
        return raw.tolist()
    return list(raw) if raw is not None else []

def compress_image(path):
    full = os.path.join(IMAGE_ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        img = Image.open(full).convert('RGB')
        img.thumbnail((IMG_MAX_SIZE, IMG_MAX_SIZE), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=IMG_QUALITY, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        print(f"    [IMG ERROR] {path}: {e}")
        return None

def extract_fields(row, field_list):
    out = {}
    for col in field_list:
        if col not in row.index:
            continue
        val = row[col]
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        s = str(val).strip()
        if s and s != 'nan':
            out[col] = s
    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ablation results...")
    with open(ABLATION_JSON) as f:
        ablation = json.load(f)

    V1_UUIDS = {
    "Bes plaque":                   "6a43e6ff-1ecc-43ec-88df-d0b91ed1530e",
    "Incense burner":               "ee7b6e1e-7e6a-46d9-aa12-0ec1857df0e9",
    "Loom weight":                  "3b9b8bbb-c037-455b-64ab-0f9d68e706d4",
    "Vessel neck":                  "7c982900-0821-4b7e-9ad1-5130a9d4004b",
    "Figurine":                     "f2df5fd8-0373-4d02-a830-87175fb3165e",
    "Seal":                         "7ecc5209-85d2-4b34-b9d5-6fd072cc7995",
    "Bulla":                        "f127f29d-5a89-45c1-9dfa-f62d989ad13b",
    "Statuette":                    "585ef91c-5ec4-4a20-cd7b-bec18b547420",
    "Ceramic lamp":                 "ff39c40a-8992-4ea4-88f6-b1a3fe626b16",
    }
    llm_by_row = {}
    for art in ablation.get('artifacts', []):
        row_idx = art.get('row_index')
        if row_idx is not None:
            llm_by_row[row_idx] = {
                'name':           art['name'],
                'dataset_source': art.get('dataset_source', 'v4'),
                'combos':         art.get('combos', []),
            }
    ablation_matrix = ablation.get('ablation_matrix', [])
    print(f"  LLM artifacts: {len(llm_by_row)}")

    print("Loading v4 parquet...")
    df_v4 = pd.read_parquet(PARQUET_V4)
    print(f"  v4: {len(df_v4)} rows, {len(df_v4.columns)} cols")

    print("Loading v1 parquet (for LLM-only artifacts)...")
    df_v1 = pd.read_parquet(PARQUET_V1)
    print(f"  v1: {len(df_v1)} rows")

    print("Loading pre-filtered artifact sample...")
    with open(ARTIFACT_SAMPLE_JSON) as f:
        artifact_sample = json.load(f)
    sample_indices = artifact_sample['row_indices']
    sample = df_v4.iloc[sample_indices].copy()
    sample['_source'] = 'v4'
    print(f"  Loaded {len(sample)} confirmed artifacts from v4")

    # force-include all LLM artifacts (v4 and v1)
    added_v4, added_v1 = 0, 0
    for row_idx, llm_info in llm_by_row.items():
        source = llm_info.get('dataset_source', 'v4')
        if source == 'v1':
            if row_idx < len(df_v1):
                row_df = df_v1.iloc[[row_idx]].copy()
                row_df['_source'] = 'v1'
                if row_idx not in sample.index:
                    sample = pd.concat([sample, row_df])
                    added_v1 += 1
            else:
                print(f"  [SKIP] v1 row {row_idx} out of v1 bounds ({len(df_v1)} rows)")
        else:
            if row_idx < len(df_v4) and row_idx not in sample.index:
                row_df = df_v4.iloc[[row_idx]].copy()
                row_df['_source'] = 'v4'
                sample = pd.concat([sample, row_df])
                added_v4 += 1
            elif row_idx >= len(df_v4):
                print(f"  [SKIP] v4 row {row_idx} out of v4 bounds ({len(df_v4)} rows)")

    sample = sample[~sample.index.duplicated(keep='first')]
    print(f"  Added {added_v4} v4 LLM artifacts, {added_v1} v1 LLM artifacts -> total {len(sample)}")

    print(f"\nProcessing {len(sample)} artifacts...")
    artifacts = []

    for i, (idx, row) in enumerate(sample.iterrows()):
        label   = str(row.get('label', '')).strip()
        project = str(row.get('project_label', '')).strip()
        llm     = llm_by_row.get(int(idx))
        source  = str(row.get('_source', 'v4'))

        print(f"  [{i+1}/{len(sample)}] {label[:40]:<40} {project[:25]}"
              + (f" [LLM/{source.upper()}]" if llm else f" [{source.upper()}]"))

        paths  = get_paths(row)
        images = []
        for p in paths:
            b64 = compress_image(p)
            if b64:
                images.append(b64)
        print(f"    {len(images)}/{len(paths)} images loaded")

        record = {
            "name":           llm['name'] if llm else label,
            "label":          label,
            "project":        project,
            "row_index":      int(idx),
            "dataset_source": llm.get('dataset_source', source) if llm else source,
            "key_fields":     extract_fields(row, KEY_FIELDS),
            "extra_fields":   extract_fields(row, EXTRA_FIELDS),
            "images":         images,
            "image_count":    len(paths),
            "has_llm":        llm is not None,
            "combos":         llm['combos'] if llm else [],
        }
        if llm and source == 'v1':
            art_name = llm['name']
            if art_name in V1_UUIDS:
                record['extra_fields']['uuid_hex'] = V1_UUIDS[art_name]
        artifacts.append(record)  # THIS WAS MISSING

    combined = {
        "artifacts":       artifacts,
        "ablation_matrix": ablation_matrix,
        "dataset_version": "v4",
        "stats": {
            "total":    len(artifacts),
            "with_llm": sum(1 for a in artifacts if a['has_llm']),
            "sample":   len(sample_indices),
        }
    }

    print(f"\nWriting {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(combined, f, indent=2)

    size_mb = os.path.getsize(OUTPUT_JSON) / 1_000_000
    print(f"  Size: {size_mb:.1f} MB")
    if size_mb > 90:
        print("  WARNING: Over 90MB — reduce IMG_MAX_SIZE or IMG_QUALITY")
    else:
        print("  Safe for GitHub Pages")

    print(f"\nDone!")
    print(f"  Total artifacts:  {len(artifacts)}")
    print(f"  With LLM outputs: {sum(1 for a in artifacts if a['has_llm'])}")
    print(f"  JSON size:        {size_mb:.1f} MB")

if __name__ == '__main__':
    main()