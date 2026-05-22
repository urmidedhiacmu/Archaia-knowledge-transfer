# run_ablation.py
#
# cd /home/udedhia/archaia_project/scripts
# source ~/archaia_env/bin/activate
# sbatch run_ablations_uses.sh

import pandas as pd
import json, os, base64, time, ast
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
OPENAI_KEY = os.getenv("ARCHAIA_OPENAI_API_KEY")
GEMINI_KEY = os.getenv("ARCHAIA_GEMINI_API_KEY")
GROQ_KEY   = False

# ── CONFIG ────────────────────────────────────────────────────────────────
PARQUET_V4 = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
PARQUET_V1 = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_text_v1.parquet'
IMAGE_ROOT = '/data/user_data/udedhia/archaia/final'
OUTPUT     = '/home/udedhia/archaia_project/uses_outputs/ablation_results.json'
MAX_IMAGES = 3
MAX_TOKENS = 700
TEMP       = 0.2

FILTERED_IMAGES_PATH = '/data/group_data/dei-group/archaia/filtered_images.json'
with open(FILTERED_IMAGES_PATH) as f:
    FILTERED_IMAGES = json.load(f)

# ── ARTIFACTS OF INTEREST ─────────────────────────────────────────────────
# parquet="v4" — artifact is in v4 (has location + date)
# parquet="v1" — artifact was filtered out of v4 (missing location or date)
#                but exists in v1; used for ablation text/image only
ARTIFACTS = [
    # confirmed v4 matches via UUID from spreadsheet
    {"name": "Reconstructed Pottery Vessel", "row": 16494,  "parquet": "v4"},
    {"name": "Stone weight",                 "row": 623,    "parquet": "v4"},
    {"name": "Coarse ware",                  "row": 18598,  "parquet": "v4"},
    {"name": "Stamp seal",                   "row": 11481,  "parquet": "v4"},
    {"name": "Bichrome sherd",               "row": 2511,   "parquet": "v4"},
    {"name": "Painted lid",                  "row": 7371,   "parquet": "v4"},
    {"name": "Lithic",                       "row": 20938,  "parquet": "v4"},
    {"name": "Greenstone axe",               "row": 11827,  "parquet": "v4"},
    {"name": "Pestle",                       "row": 7098,   "parquet": "v4"},
    {"name": "Tube",                         "row": 16066,  "parquet": "v4"},
    {"name": "Stone vessel",                 "row": 1097,   "parquet": "v4"},
    # v1 only — filtered out of v4 (no location or date in OpenContext)
    {"name": "Bes plaque",                   "row": 6593,   "parquet": "v1"},
    {"name": "Incense burner",               "row": 30766,  "parquet": "v1"},
    {"name": "Loom weight",                  "row": 14741,  "parquet": "v1"},
    {"name": "Vessel neck",                  "row": 23811,  "parquet": "v1"},
    {"name": "Figurine",                     "row": 18200,  "parquet": "v1"},
    {"name": "Seal",                         "row": 6,      "parquet": "v1"},
    {"name": "Bulla",                        "row": 30919,  "parquet": "v1"},
    {"name": "Statuette",                    "row": 216,    "parquet": "v1"},
    {"name": "Ceramic lamp",                 "row": 3212,   "parquet": "v1"},
]

# ── PROMPTS ───────────────────────────────────────────────────────────────
PROMPTS = {

"archaeological": """You are writing a use-hypothesis section for an archaeological site report.
Based on the artifact data provided, produce:
1. One sentence identifying the artifact and its key physical characteristics
2. Three to five numbered use hypotheses, each with:
   - A specific proposed function
   - Evidence from the provided data that supports it
   - A confidence level (high / medium / low)
3. A one-sentence note on what additional evidence would strengthen or refute the leading hypothesis
Use precise academic language. Hedge all claims appropriately. Do not invent facts not in the data. Under 320 words.""",

"museum_label": """You are writing interpretive content for a museum exhibit on ancient material culture.
Based on the artifact data provided, write:
1. A short engaging title for this artifact (max 6 words)
2. A 2-3 sentence description for a general adult audience — vivid, concrete, no jargon
3. A section called "How might this have been used?" with 3-4 plausible uses as short accessible sentences
4. One sentence connecting this object to a universal human experience (cooking, trade, ritual, status, etc.)
Make it engaging and human. Be honest about uncertainty. Under 280 words.""",

"generative": """You are reconstructing the human story behind an archaeological artifact.
Based on the data provided, write a short first-person narrative (150-200 words) imagining:
- Who likely owned or used this object (their role, status, or daily life in this culture and period)
- What they used it for, based specifically on the physical evidence provided
- How and why it ended up in the archaeological context where it was found
Then add 2-3 bullet points labeling which specific data fields informed each element of your story.
Flag any detail you invented without evidence."""

}

INPUT_MODES = ["sparse", "full_and_images"]

SPARSE_FIELDS = [
    ('label','Label'), ('project_label','Project'),
    ('recovered_material','Material'), ('recovered_object_type','Object Type'),
    ('recovered_size','Size'), ('recovered_condition','Condition'),
    ('recovered_period','Period'), ('earliest','Earliest Date'), ('latest','Latest Date'),
]

FULL_FIELDS = [
    ('label','Label'), ('project_label','Project'),
    ('metadata','Project Metadata'),
    ('recovered_artifact_name','Artifact Name'),
    ('recovered_material','Material'), ('recovered_material_note','Material Note'),
    ('recovered_object_type','Object Type'), ('recovered_object_type_note','Object Type Note'),
    ('recovered_period','Period'), ('recovered_chronotype','Chronotype'),
    ('recovered_condition','Condition'), ('recovered_size','Size'),
    ('recovered_munsell_color','Munsell Color'), ('recovered_munsell_number','Munsell Number'),
    ('recovered_fabric_description','Fabric'), ('recovered_fabric_group','Fabric Group'),
    ('recovered_decorative_technique','Decorative Technique'),
    ('recovered_description','Description'), ('recovered_description_remarks','Description Remarks'),
    ('recovered_note','Note'), ('recovered_function','Recorded Function'),
    ('recovered_location','Location'), ('recovered_specific_context','Specific Context'),
    ('recovered_specific_location','Specific Location'),
    ('recovered_locus','Locus'), ('recovered_locus_id','Locus ID'),
    ('recovered_registration_date','Registration Date'), ('recovered_disposition','Disposition'),
    ('earliest','Earliest Date'), ('latest','Latest Date'),
    ('latitude','Latitude'), ('longitude','Longitude'),
]

# ── HELPERS ───────────────────────────────────────────────────────────────
def build_text(row, fields, include_json=False):
    lines = []
    for col, label in fields:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            lines.append(f"{label}: {str(val).strip()[:400]}")
    if include_json:
        raw = row.get('recovered_text_fields_json')
        if pd.notna(raw):
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                lines.append("\nFull recovered metadata:")
                for k, v in parsed.items():
                    val_str = ', '.join(v) if isinstance(v, list) else str(v)
                    lines.append(f"  {k}: {val_str[:300]}")
            except Exception:
                pass
    return '\n'.join(lines) or '(No text metadata available)'

def get_input(row, mode, images):
    if mode == 'sparse':
        return build_text(row, SPARSE_FIELDS, include_json=False), []
    else:
        return build_text(row, FULL_FIELDS, include_json=True), images

def get_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try:    return json.loads(raw)
        except: return ast.literal_eval(raw)
    return list(raw) if raw is not None else []

def load_images(paths):
    out = []
    for p in paths[:MAX_IMAGES]:
        full = os.path.join(IMAGE_ROOT, p)
        if not os.path.exists(full):
            continue
        with open(full, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png'}.get(
            Path(full).suffix.lower().lstrip('.'), 'jpeg')
        out.append((f"data:image/{mime};base64,{data}", mime, data))
    return out

# ── API CALLS ─────────────────────────────────────────────────────────────
def call_openai(client, system, text, images):
    content = []
    if text: content.append({"type": "text", "text": text})
    for uri, _, _ in images:
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
    if not content: content.append({"type": "text", "text": "(No input)"})
    r = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=MAX_TOKENS, temperature=TEMP,
        messages=[{"role":"system","content":system},{"role":"user","content":content}])
    return r.choices[0].message.content.strip()

def call_gemini(system, text, images):
    model = genai.GenerativeModel(
        model_name="gemini-flash-lite-latest",
        system_instruction=system
    )
    parts = []
    if text: parts.append(text)
    for _, mime, data in images:
        parts.append({"mime_type": f"image/{mime}", "data": base64.b64decode(data)})
    if not parts: parts.append("(No input)")
    r = model.generate_content(
        parts,
        generation_config={"temperature": TEMP, "max_output_tokens": MAX_TOKENS}
    )
    return r.text.strip()

def call_groq(client, system, text, images):
    user_text = text or "(No input)"
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=MAX_TOKENS, temperature=TEMP,
        messages=[{"role":"system","content":system},{"role":"user","content":user_text}])
    return r.choices[0].message.content.strip()

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    oai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
    genai.configure(api_key=GEMINI_KEY)
    groq_client = None
    if GROQ_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_KEY)

    active_models = []
    if oai_client:  active_models.append("gpt-4o-mini");             print("GPT-4o-mini ready")
    if GEMINI_KEY:  active_models.append("gemini-flash-lite-latest"); print("Gemini Flash Lite ready")
    if groq_client: active_models.append("llama-3.3-70b-versatile"); print("Groq Llama ready")
    if not active_models:
        print("ERROR: No API keys found"); return

    print('\nLoading parquets...')
    df_v4 = pd.read_parquet(PARQUET_V4)
    df_v1 = pd.read_parquet(PARQUET_V1)
    print(f'v4: {len(df_v4)} rows  |  v1: {len(df_v1)} rows\n')

    print("Resolving artifact rows...")
    resolved = []
    for art in ARTIFACTS:
        df     = df_v4 if art['parquet'] == 'v4' else df_v1
        idx    = art['row']
        if idx >= len(df):
            print(f"  [WARN] row {idx} out of range for '{art['name']}' in {art['parquet']} — skipping")
            continue
        row    = df.iloc[idx]
        label  = str(row.get('label', '')).strip()
        proj   = str(row.get('project_label', '')).strip()
        src    = art['parquet'].upper()
        print(f"  [{src}] {art['name']:<30} row={idx:<6} label={label[:30]}  project={proj[:30]}")
        resolved.append({**art, "_row_data": row})

    print(f"\nResolved {len(resolved)}/{len(ARTIFACTS)} artifacts\n")

    cache, all_results = {}, []
    unique_rows  = len(set((a['row'], a['parquet']) for a in resolved))
    total_calls  = unique_rows * len(active_models) * len(INPUT_MODES) * len(PROMPTS)
    done         = 0

    for art in resolved:
        row      = art["_row_data"]
        row_idx  = art["row"]
        source   = art["parquet"]
        filtered = FILTERED_IMAGES.get(art['name'], {})
        paths    = filtered.get('kept', get_paths(row))
        images   = load_images(paths)
        thumb    = images[0][0] if images else None

        print(f'-- {art["name"]}  [{source.upper()}]  (row={row_idx}, images={len(images)}/{len(paths)})')

        combos = []
        for mode in INPUT_MODES:
            text, imgs = get_input(row, mode, images)
            for pname, psystem in PROMPTS.items():
                for model in active_models:
                    cache_key = (row_idx, source, model, mode, pname)
                    if cache_key in cache:
                        response = cache[cache_key]
                        print(f'   [cached] {model} | {mode} | {pname}')
                    else:
                        try:
                            if model == "gpt-4o-mini":
                                response = call_openai(oai_client, psystem, text, imgs)
                            elif model == "gemini-flash-lite-latest":
                                response = call_gemini(psystem, text, imgs)
                            elif model == "llama-3.3-70b-versatile":
                                response = call_groq(groq_client, psystem, text, imgs)
                            else:
                                response = "[SKIPPED: unknown model]"
                        except Exception as e:
                            response = f'[ERROR: {e}]'
                        cache[cache_key] = response
                        done += 1
                        ok = 'ok' if not response.startswith('[') else response[:60]
                        print(f'   {done}/{total_calls} {model} | {mode} | {pname} -> {ok}')
                        time.sleep(1.2)

                    combos.append({
                        "model":        model,
                        "input_mode":   mode,
                        "prompt_style": pname,
                        "response":     response,
                    })

        skip = {'image_paths','geometry','metadata','recovered_text_fields_json'}
        text_fields = {
            k: str(row[k]) for k in row.index
            if k not in skip and pd.notna(row[k]) and str(row[k]).strip()
        }

        all_results.append({
            "name":           art['name'],
            "row_index":      row_idx,
            "dataset_source": source,   # "v4" or "v1" — used by viewer for badge
            "text_fields":    text_fields,
            "thumbnail":      thumb,
            "image_count":    len(paths),
            "combos":         combos,
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump({
            "artifacts": all_results,
            "ablation_matrix": [
                {"model": m, "input_mode": im, "prompt_style": ps}
                for im in INPUT_MODES
                for ps in PROMPTS
                for m in active_models
            ]
        }, f, indent=2)

    print(f'\nDone -> {OUTPUT}')
    print(f'Total API calls: {done}')

if __name__ == '__main__':
    main()