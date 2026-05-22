# filter_images.py
#
# Classifies all images for the 20 artifacts using Gemini
# and saves a filtered image mapping to filtered_images.json
#
# cd /home/udedhia/archaia_project/scripts
# source ~/archaia_env/bin/activate
# python3 filter_images.py

import os, json, base64, time, ast
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
genai.configure(api_key=os.getenv("ARCHAIA_GEMINI_API_KEY"))

PARQUET   = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet'
IMAGE_DIR = '/data/group_data/dei-group/archaia'
OUTPUT    = '/data/group_data/dei-group/archaia/filtered_images.json'

ARTIFACTS = [
    {"name": "Reconstructed Pottery Vessel", "row": 20525},
    {"name": "Bes plaque",                   "row": 6593},
    {"name": "Incense burner",               "row": 30766},
    {"name": "Loom weight",                  "row": 14741},
    {"name": "Stone weight",                 "row": 30710},
    {"name": "Coarse ware",                  "row": 11493},
    {"name": "Vessel neck",                  "row": 23811},
    {"name": "Figurine",                     "row": 18200},
    {"name": "Seal",                         "row": 6},
    {"name": "Stamp seal",                   "row": 6},
    {"name": "Bichrome sherd",               "row": 8006},
    {"name": "Painted lid",                  "row": 12949},
    {"name": "Bulla",                        "row": 30919},
    {"name": "Statuette",                    "row": 216},
    {"name": "Lithic",                       "row": 3637},
    {"name": "Greenstone axe",               "row": 16501},
    {"name": "Ceramic lamp",                 "row": 3212},
    {"name": "Pestle",                       "row": 5374},
    {"name": "Tube",                         "row": 233},
    {"name": "Stone vessel",                 "row": 11202},
]

SYSTEM = """You are an archaeological image classifier.
You will be shown an image. Respond with ONLY one word:
- "artifact" if the image shows an archaeological object, artifact, pottery, tool, figurine, seal, or any man-made object being studied
- "other" if the image shows a landscape, excavation site, field, document, map, scale bar only, person, aerial view, or anything that is not the artifact itself

No explanation. One word only: artifact or other"""

def get_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try:    return json.loads(raw)
        except: return ast.literal_eval(raw)
    return list(raw)

def load_image_b64(path):
    full = os.path.join(IMAGE_DIR, path)
    if not os.path.exists(full):
        return None, None
    with open(full, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png'}.get(
        Path(full).suffix.lower().lstrip('.'), 'jpeg')
    return data, mime

def classify_image(data, mime):
    model = genai.GenerativeModel("gemini-flash-lite-latest", system_instruction=SYSTEM)
    try:
        r = model.generate_content([
            {"mime_type": f"image/{mime}", "data": base64.b64decode(data)},
            "Is this an artifact image or something else?"
        ])
        result = r.text.strip().lower()
        return "artifact" if "artifact" in result else "other"
    except Exception as e:
        print(f"    ERROR: {e}")
        return "error"

def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} rows\n")

    model = genai.GenerativeModel("gemini-flash-lite-latest", system_instruction=SYSTEM)
    results = {}
    seen_rows = set()

    for art in ARTIFACTS:
        row_idx = art['row']

        # Skip duplicate rows (seal/stamp seal)
        if row_idx in seen_rows:
            print(f"── {art['name']} (row={row_idx}) [using cached result]")
            results[art['name']] = results[[a['name'] for a in ARTIFACTS if a['row']==row_idx and a['name'] in results][0]]
            continue
        seen_rows.add(row_idx)

        row   = df.iloc[row_idx]
        paths = get_paths(row)
        print(f"── {art['name']}  (row={row_idx}, total images={len(paths)})")

        kept, rejected = [], []

        for p in paths:
            data, mime = load_image_b64(p)
            if data is None:
                print(f"   [MISSING] {p}")
                continue

            label = classify_image(data, mime)
            if label == "artifact":
                kept.append(p)
                print(f"   ✓ artifact  {Path(p).name}")
            elif label == "other":
                rejected.append(p)
                print(f"   ✗ other     {Path(p).name}")
            else:
                print(f"   ? error     {Path(p).name}")

            time.sleep(0.8)  # rate limit

        print(f"   → kept {len(kept)}/{len(paths)}\n")
        results[art['name']] = {
            "row_index": row_idx,
            "total":     len(paths),
            "kept":      kept,
            "rejected":  rejected,
        }

    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved to {OUTPUT}")

    # Summary
    print("\n── SUMMARY ──")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"  {name}: {len(r['kept'])}/{r['total']} kept")

if __name__ == '__main__':
    main()