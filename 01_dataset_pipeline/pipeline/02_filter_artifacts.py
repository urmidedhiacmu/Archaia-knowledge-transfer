# filter_artifacts.py
#
# Finds 100 confirmed artifacts from the v4 augmented dataset.
# v4 is already filtered by item_class (no loci/trenches/sites),
# so stage 1 rule filter is a lighter safety net only.
#
# cd /home/udedhia/archaia_project/scripts
# source ~/archaia_env/bin/activate
# python3 -u filter_artifacts.py

import os, json, time, re
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
genai.configure(api_key=os.getenv("ARCHAIA_GEMINI_API_KEY"))

PARQUET         = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
OUTPUT          = '/home/udedhia/archaia_project/gh_deploy/artifact_sample_100.json'
TARGET_N        = 100
POOL_SIZE       = 400
PER_COMBO_CAP  = 12   # was 2 — more candidates per (project, class) pair
PER_CLASS_CAP  = 20   # was 15 — slightly tighter so more classes represented
PER_PROJECT_CAP = 10   # was 6 — slightly tighter so more projects represented   

# v4 already removed Locus/Survey Unit/Site/Trench/Context via item_class.
# These patterns catch residual ambiguous labels within retained classes.
NON_ARTIFACT_PATTERNS = [
    r'\bbatch\b', r'\bwall\b', r'\bmosque\b',
    r'\broom\b', r'\bbuilding\b', r'\bfloor\b',
    r'\bpit\b', r'\bdump\b', r'\bfill\b',
    r'\bsediment\b', r'\bsoil\b', r'\bash\b',
    r'\bdeposit\b', r'\blevel\b', r'\bphase\b',
    r'\btomb\b', r'\bgrave\b',
]

NON_ARTIFACT_OBJECT_TYPES = [
    'organic (ecofact)', 'ecofact', 'production waste',
]

ARTIFACT_PATTERNS = [
    r'\bsherd\b', r'\bvessel\b', r'\bfigurine\b', r'\bseal\b',
    r'\bstatuette\b', r'\bbulla\b', r'\blamp\b', r'\bpestle\b',
    r'\baxe\b', r'\bblade\b', r'\bpin\b', r'\bbead\b',
    r'\bamulet\b', r'\bpendant\b', r'\bring\b', r'\bbracelet\b',
    r'\bknife\b', r'\bsickle\b', r'\barrowhead\b', r'\bscraper\b',
    r'\bweight\b', r'\bspindle\b', r'\bwhorl\b', r'\bneedle\b',
    r'\bjug\b', r'\bjar\b', r'\bbowl\b', r'\bcup\b', r'\bplate\b',
    r'\btile\b', r'\bplaque\b', r'\btoken\b', r'\bcoin\b',
]

def stage1_classify(row):
    label    = str(row.get('label', '')).lower()
    obj_type = str(row.get('recovered_object_type', '') or '').lower()

    for pat in NON_ARTIFACT_OBJECT_TYPES:
        if pat in obj_type:
            return 'not_artifact'
    for pat in NON_ARTIFACT_PATTERNS:
        if re.search(pat, label):
            return 'not_artifact'
    for pat in ARTIFACT_PATTERNS:
        if re.search(pat, label):
            return 'artifact'
    if obj_type and obj_type not in ['none', 'nan', '']:
        is_bad = any(b in obj_type for b in NON_ARTIFACT_OBJECT_TYPES)
        if not is_bad:
            return 'artifact'
    return 'uncertain'

GEMINI_SYSTEM = """You are classifying archaeological dataset records.
Given a label, object type, and description, respond with ONLY one word:
- "artifact" if this record represents a physical man-made object (pottery, tool, figurine, seal, ornament, weapon, coin, tile, etc.)
- "not_artifact" if this record represents a context, location, or non-object (locus, trench, feature, floor, wall, layer, deposit, soil sample, batch, ecofact, etc.)
No explanation. One word only: artifact or not_artifact"""

def stage2_gemini(row):
    model = genai.GenerativeModel("gemini-flash-lite-latest", system_instruction=GEMINI_SYSTEM)
    label    = str(row.get('label', '')).strip()
    obj_type = str(row.get('recovered_object_type', '') or '').strip()
    desc     = str(row.get('recovered_description', '') or '').strip()[:300]
    note     = str(row.get('recovered_note', '') or '').strip()[:200]
    prompt   = f"Label: {label}\nObject Type: {obj_type}\nDescription: {desc}\nNote: {note}"
    try:
        r = model.generate_content(prompt)
        result = r.text.strip().lower()
        return 'artifact' if 'artifact' in result and 'not' not in result else 'not_artifact'
    except Exception as e:
        print(f"    [GEMINI ERROR] {e}")
        return 'uncertain'

def main():
    print("Loading v4 parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"  Total rows: {len(df)}")
    if 'item_class_label' in df.columns:
        print(f"  item_class breakdown:\n{df['item_class_label'].value_counts().to_string()}")

    rcols = [c for c in df.columns if c.startswith('recovered_')
             and c != 'recovered_text_fields_json']
    df['_richness'] = df[rcols].notna().sum(axis=1)

    # v4 is already item_class filtered — every row is a legitimate artifact class.
    # Sample directly for diversity: across projects AND item_class.
    # Strategy: for each (project, class) pair, take the richest rows up to cap.
    print("\nSampling by project x class diversity...")

    # Only use classes that make sense as portable artifacts for the viewer
    # Feature and Structure are kept in v4 but deprioritised here
    CLASS_PRIORITY = [
        'Object', 'Pottery', 'Coin', 'Architectural Element',
        'Lithic', 'Animal Bone', 'Groundstone', 'Sculpture',
        'Glass', 'Shell', 'Human Bone', 'Biological record',
        'Sample', 'Feature', 'Structure'
    ]

    # Build candidate pool: richest rows per (project, class) combo
    frames = []
    for cls in CLASS_PRIORITY:
        cls_df = df[df['item_class_label'] == cls] if 'item_class_label' in df.columns else df
        for proj, grp in cls_df.groupby('project_label'):
            best = grp.sort_values('_richness', ascending=False).head(PER_COMBO_CAP)
            frames.append(best)

    candidate_pool = pd.concat(frames)
    candidate_pool = candidate_pool[~candidate_pool.index.duplicated(keep='first')]
    print(f"  Candidate pool: {len(candidate_pool)} rows from "
          f"{candidate_pool['project_label'].nunique()} projects")

    # Run Gemini only on truly ambiguous labels (uncertain stage1)
    print("\nStage 1: Quick label check on candidates...")
    candidate_pool['_stage1'] = candidate_pool.apply(stage1_classify, axis=1)
    uncertain_cands = candidate_pool[candidate_pool['_stage1'] == 'uncertain']
    print(f"  Uncertain in pool: {len(uncertain_cands)}")

    gemini_results = {}
    if len(uncertain_cands) > 0:
        print(f"Stage 2: Gemini classifying {len(uncertain_cands)} uncertain candidates...")
        for i, (idx, row) in enumerate(uncertain_cands.iterrows()):
            result = stage2_gemini(row)
            gemini_results[idx] = result
            status = 'artifact' if result == 'artifact' else 'not_artifact'
            print(f"  [{i+1}/{len(uncertain_cands)}] {str(row.get('label',''))[:40]:<40} {status}")
            time.sleep(0.6)

    # Remove confirmed non-artifacts
    def is_artifact(idx, row):
        s1 = row['_stage1']
        if s1 == 'not_artifact': return False
        if s1 == 'uncertain':    return gemini_results.get(idx, 'uncertain') == 'artifact'
        return True  # 'artifact'

    candidate_pool['_keep'] = [
        is_artifact(idx, row) for idx, row in candidate_pool.iterrows()
    ]
    candidate_pool = candidate_pool[candidate_pool['_keep']].copy()
    candidate_pool = candidate_pool.sort_values('_richness', ascending=False)
    print(f"  After filtering: {len(candidate_pool)} candidates")

    # Final selection with class + project caps
    print(f"\nSelecting final {TARGET_N} artifacts...")
    final_rows = []
    class_counts   = {}
    project_counts = {}

    for idx, row in candidate_pool.iterrows():
        cls  = str(row.get('item_class_label', 'unknown'))
        proj = str(row.get('project_label', 'unknown'))
        if class_counts.get(cls, 0)   >= PER_CLASS_CAP:    continue
        if project_counts.get(proj, 0) >= PER_PROJECT_CAP: continue
        final_rows.append(int(idx))
        class_counts[cls]   = class_counts.get(cls, 0) + 1
        project_counts[proj] = project_counts.get(proj, 0) + 1
        if len(final_rows) >= TARGET_N:
            break

    print(f"  Final: {len(final_rows)} artifacts")
    print(f"\n  Class distribution:")
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<30} {n}")
    print(f"\n  Project distribution (top 15):")
    for proj, n in sorted(project_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {proj:<50} {n}")

    print(f"\nSample (first 10):")
    for idx in final_rows[:10]:
        row = df.iloc[idx]
        cls = str(row.get('item_class_label', ''))
        print(f"  [{idx}] {str(row['label'])[:35]:<35} | {str(row['project_label'])[:30]:<30} | {cls:<25} | richness={int(row['_richness'])}")

    output = {
        "row_indices": final_rows,
        "total": len(final_rows),
        "dataset": "archaia_final_dataset_augmented_v4.parquet",
        "stats": {
            "class_distribution":   class_counts,
            "project_distribution": project_counts,
            "projects_represented": len(project_counts),
        }
    }
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(final_rows)} row indices to {OUTPUT}")

if __name__ == '__main__':
    main()