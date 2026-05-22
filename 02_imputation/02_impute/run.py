import os, json, base64, requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from openai import OpenAI

INPUT_PATH  = '/home/udedhia/archaia_project/archaia_impute/data/imputation_input.parquet'
V4_PATH     = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
INDEX_DIR   = '/home/udedhia/archaia_project/archaia_impute/index'
OUTPUT_DIR  = '/home/udedhia/archaia_project/archaia_impute/outputs'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'imputed.parquet')
VOCAB_PATH = '/home/udedhia/archaia_project/archaia_impute/data/vocab.json'
IMPUTE_FIELDS = [
    'recovered_material', 'recovered_object_type', 'recovered_condition',
    'recovered_period', 'recovered_description'
]

API_KEY      = os.environ.get('ARCHAIA_OPENAI_API_KEY', '')
TOP_K        = 50
MAX_ARTIFACTS = 500
IMG_DIM      = 1024
TEXT_DIM     = 384

def fetch_and_encode_image(url, processor, dino, device):
    import torch
    try:
        r = requests.get(url, timeout=8)
        img = Image.open(BytesIO(r.content)).convert('RGB')
        inputs = processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            out = dino(**inputs)
        return out.last_hidden_state[:,0,:].cpu().numpy().flatten()
    except:
        return None

def fetch_image_b64(url, max_size=512):
    try:
        r = requests.get(url, timeout=8)
        img = Image.open(BytesIO(r.content)).convert('RGB')
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

def encode_text(row, text_model):
    parts = []
    for f in IMPUTE_FIELDS:
        v = row.get(f)
        if v and str(v) not in ('nan','None',''):
            parts.append(f"{f.replace('recovered_','')}: {str(v)[:200]}")
    parts.append(f"class: {row.get('item_class_label','')}")
    text = ' | '.join(parts) if parts else str(row.get('label',''))
    return text_model.encode([text], show_progress_bar=False)[0]

def get_vocab_for_artifact(vocab, field, item_class, project):
    fv = vocab.get(field, {})
    if project in fv.get('_by_project', {}):
        return fv['_by_project'][project]
    if item_class in fv.get('_by_class', {}):
        return fv['_by_class'][item_class]
    return fv.get('_global', [])

def build_prompt(artifact, neighbors, missing_fields, vocab):
    available = {}
    for f in ['item_class_label','label','recovered_material','recovered_object_type',
              'recovered_condition','recovered_period','recovered_description']:
        v = artifact.get(f,'')
        if v and str(v) not in ('nan','None','') and f not in missing_fields:
            available[f] = str(v)

    neighbor_ctx = []
    for n in neighbors[:TOP_K]:
        entry = {k: n[k] for k in ['label','item_class','project'] if n.get(k)}
        for f in IMPUTE_FIELDS:
            if n.get(f,'') not in ('','nan','None'):
                entry[f] = n[f]
        neighbor_ctx.append(entry)

    constraints = get_vocab_for_artifact(vocab, target_field,
    row.get('item_class_label',''), row.get('project_label',''))[:20]

    prompt = f"""You are an expert archaeologist filling missing metadata for an artifact.

ARTIFACT KNOWN FIELDS:
{json.dumps(available, indent=2)}

SIMILAR ARTIFACTS FROM DATABASE:
{json.dumps(neighbor_ctx, indent=2)}

MISSING FIELDS TO FILL: {missing_fields}

ALLOWED VALUES (pick from these lists where provided):
{json.dumps(constraints, indent=2)}

Rules:
- Values MUST come from the allowed lists above where provided
- Use the artifact image and similar artifacts to inform your answer
- For recovered_description write a brief factual description based on what you can observe
- If genuinely unknown output null
- No explanation, output ONLY valid JSON

Output format:
{json.dumps({f: "value or null" for f in missing_fields}, indent=2)}"""
    return prompt

def main():
    import torch, faiss
    from transformers import AutoImageProcessor, AutoModel
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = OpenAI(api_key=API_KEY)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Loading models...')
    processor  = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    dino       = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    dino.eval()
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print('Loading index...')
    index = faiss.read_index(os.path.join(INDEX_DIR, 'faiss.index'))
    with open(os.path.join(INDEX_DIR, 'id_map.json')) as f:
        id_map = {int(k): v for k, v in json.load(f).items()}

    print('Loading data...')
    df_input = pd.read_parquet(INPUT_PATH)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    subset  = df_input.head(MAX_ARTIFACTS)
    results = []
    print(f'Imputing {len(subset):,} artifacts...')

    for i, (_, row) in enumerate(subset.iterrows()):
        print(f'[{i+1}/{len(subset)}] {str(row.get("label",""))[:40]}')

        missing = [
            f for f in IMPUTE_FIELDS
            if pd.isna(row.get(f)) or str(row.get(f,'')) in ('','nan','None')
        ]
        if not missing:
            print('  nothing to impute, skipping')
            continue

        # get image URLs
        urls = row.get('image_urls', [])
        if isinstance(urls, str):
            try: urls = json.loads(urls)
            except: urls = []

        # encode image
        img_emb = None
        img_b64 = None
        for url in urls[:3]:
            img_emb = fetch_and_encode_image(url, processor, dino, device)
            if img_emb is not None:
                img_b64 = fetch_image_b64(url)
                break
        if img_emb is None:
            img_emb = np.zeros(IMG_DIM, dtype=np.float32)

        # encode text
        text_emb = encode_text(row.to_dict(), text_model)
        combined = np.concatenate([img_emb, text_emb]).astype(np.float32)
        norm = np.linalg.norm(combined)
        if norm > 0: combined = combined / norm

        # retrieve neighbors that have at least one of the missing fields
        _, I = index.search(combined.reshape(1,-1), TOP_K * 3)
        neighbors = []
        for idx in I[0]:
            entry = id_map.get(int(idx))
            if entry and any(f in entry.get('complete_fields',[]) for f in missing):
                neighbors.append(entry)
            if len(neighbors) >= TOP_K:
                break

        # build prompt and call GPT-4o
        prompt = build_prompt(row.to_dict(), neighbors, missing, vocab)
        messages = [{'role': 'user', 'content': []}]
        if img_b64:
            messages[0]['content'].append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{img_b64}', 'detail': 'low'}
            })
        messages[0]['content'].append({'type': 'text', 'text': prompt})

        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                max_tokens=400,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace('```json','').replace('```','').strip()
            parsed = json.loads(raw)
        except Exception as e:
            print(f'  GPT error: {e}')
            parsed = {}

        result = {'uuid_hex': row['uuid_hex'], 'label': row.get('label',''),
                  'item_class_label': row.get('item_class_label','')}
        for f in missing:
            val = parsed.get(f)
            result[f + '_imputed']     = val
            result[f + '_imputed_src'] = 'gpt4o+dinov2'
            result[f + '_imputed_conf']= 'medium' if val and val != 'null' else None
        results.append(result)
        print(f'  missing={missing}')
        print(f'  imputed={parsed}')

    results_df = pd.DataFrame(results)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f'\nWrote {OUTPUT_PATH} with {len(results_df):,} rows')

if __name__ == '__main__':
    main()
