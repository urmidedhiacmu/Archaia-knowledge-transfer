# run_eval.py — uses v4_eval.parquet (genuinely held out from index)

import os, json, ast, base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from openai import OpenAI

TRAIN_PATH  = '/home/udedhia/archaia_project/archaia_impute/data/v4_train.parquet'
EVAL_PATH   = '/home/udedhia/archaia_project/archaia_impute/data/v4_eval.parquet'
VOCAB_PATH  = '/home/udedhia/archaia_project/archaia_impute/data/vocab.json'
IMAGE_ROOT  = '/data/user_data/udedhia/archaia/final'
INDEX_DIR   = '/home/udedhia/archaia_project/archaia_impute/index'
OUTPUT_DIR  = '/home/udedhia/archaia_project/archaia_impute/outputs'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'eval_results_top50.json')

IMPUTE_FIELDS      = ['recovered_material', 'recovered_object_type',
                      'recovered_condition', 'recovered_period', 'recovered_description']
CATEGORICAL_FIELDS = ['recovered_material', 'recovered_object_type',
                      'recovered_condition', 'recovered_period']

API_KEY     = os.environ.get('ARCHAIA_OPENAI_API_KEY', '')
TOP_K       = 50
N_PER_FIELD = 50
IMG_DIM     = 1024
TEXT_DIM    = 384

def get_image_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try: return json.loads(raw)
        except:
            try: return ast.literal_eval(raw)
            except: return []
    if hasattr(raw, 'tolist'): return raw.tolist()
    return list(raw) if raw is not None else []

def encode_image_from_disk(path, processor, dino, device):
    import torch
    try:
        full   = os.path.join(IMAGE_ROOT, path)
        img    = Image.open(full).convert('RGB')
        inputs = processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            out = dino(**inputs)
        return out.last_hidden_state[:,0,:].cpu().numpy().flatten()
    except:
        return None

def fetch_image_b64(path, max_size=512):
    try:
        full = os.path.join(IMAGE_ROOT, path)
        img  = Image.open(full).convert('RGB')
        img.thumbnail((max_size, max_size))
        buf  = BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None

def encode_text(row, text_model, blank_field):
    parts = []
    for f in IMPUTE_FIELDS:
        if f == blank_field: continue
        v = row.get(f)
        if v and str(v) not in ('nan','None',''):
            parts.append(f"{f.replace('recovered_','')}: {str(v)[:200]}")
    parts.append(f"class: {row.get('item_class_label','')}")
    parts.append(f"project: {row.get('project_label','')}")
    text = ' | '.join(parts) if parts else str(row.get('label',''))
    return text_model.encode([text], show_progress_bar=False)[0]

def get_vocab_for_artifact(vocab, field, item_class, project):
    # priority: project > class > global
    fv = vocab.get(field, {})
    if project in fv.get('_by_project', {}):
        return fv['_by_project'][project]
    if item_class in fv.get('_by_class', {}):
        return fv['_by_class'][item_class]
    return fv.get('_global', [])

def build_prompt(artifact_info, neighbors, target_field, vocab):
    available = {}
    for f in IMPUTE_FIELDS:
        if f == target_field: continue
        v = artifact_info.get(f,'')
        if v and str(v) not in ('nan','None',''):
            available[f] = str(v)
    available['item_class'] = artifact_info.get('item_class_label','')
    available['label']      = artifact_info.get('label','')
    available['project']    = artifact_info.get('project_label','')

    neighbor_ctx = []
    for n in neighbors[:TOP_K]:
        entry = {k: n[k] for k in ['label','item_class','project'] if n.get(k)}
        if n.get(target_field,'') not in ('','nan','None'):
            entry[target_field] = n[target_field]
        neighbor_ctx.append(entry)

    constraints = get_vocab_for_artifact(
        vocab, target_field,
        artifact_info.get('item_class_label',''),
        artifact_info.get('project_label','')
    )[:20] if target_field in CATEGORICAL_FIELDS else []

    if target_field in CATEGORICAL_FIELDS:
        out_fmt  = f'{{"{target_field}": "value or null", "{target_field}_top3": ["val1","val2","val3"]}}'
        task     = f"Fill the missing field. Also provide top 3 candidates in {target_field}_top3."
        con_str  = f"ALLOWED VALUES: {json.dumps(constraints)}" if constraints else ""
    else:
        out_fmt  = f'{{"{target_field}": "your description"}}'
        task     = "Write a brief factual description based on image and similar artifacts."
        con_str  = ""

    return f"""You are an expert archaeologist filling a missing metadata field.

ARTIFACT KNOWN FIELDS:
{json.dumps(available, indent=2)}

SIMILAR ARTIFACTS FROM DATABASE:
{json.dumps(neighbor_ctx, indent=2)}

TASK: {task}
MISSING FIELD: {target_field}
{con_str}

Rules:
- Values MUST come from allowed list if provided
- Use image and similar artifacts
- Output ONLY valid JSON

Output: {out_fmt}"""

def compute_metrics(pred, gt, target_field, sem_model):
    from rapidfuzz import fuzz
    metrics = {}
    metrics['exact_match'] = (
        str(pred).strip().lower() == str(gt).strip().lower()
        if pred and gt else False
    )
    metrics['fuzzy_token_sort'] = (
        fuzz.token_sort_ratio(str(pred), str(gt)) / 100.0
        if pred and gt else 0.0
    )
    metrics['fuzzy_partial'] = (
        fuzz.partial_ratio(str(pred), str(gt)) / 100.0
        if pred and gt else 0.0
    )
    if pred and gt:
        embs = sem_model.encode([str(pred), str(gt)])
        cos  = float(np.dot(embs[0], embs[1]) /
                     (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8))
        metrics['semantic_sim'] = round(cos, 4)
    else:
        metrics['semantic_sim'] = 0.0
    if target_field == 'recovered_description' and pred and gt:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref  = [str(gt).lower().split()]
        hyp  = str(pred).lower().split()
        metrics['bleu'] = round(
            sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1), 4
        )
    return metrics

def main():
    import torch, faiss
    from transformers import AutoImageProcessor, AutoModel
    from sentence_transformers import SentenceTransformer

    os.system('pip install rapidfuzz nltk -q')
    import nltk; nltk.download('punkt', quiet=True)

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

    print('Loading vocab...')
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    print('Loading eval split...')
    df_eval = pd.read_parquet(EVAL_PATH)
    print(f'  {len(df_eval):,} eval artifacts')

    all_results = {}
    summary     = {}

    for target_field in IMPUTE_FIELDS:
        print(f'\n{"="*60}')
        print(f'Evaluating: {target_field}')
        print(f'{"="*60}')

        has_field = df_eval[df_eval[target_field].notna()].copy()
        has_field = has_field[~has_field[target_field].astype(str).isin(['nan','None',''])]
        print(f'  {len(has_field):,} eval artifacts have this field')

        eval_set = has_field.sample(n=min(N_PER_FIELD, len(has_field)), random_state=42)
        print(f'  Evaluating on {len(eval_set):,}')

        field_results = []
        metric_accum  = {}

        for i, (_, row) in enumerate(eval_set.iterrows()):
            print(f'  [{i+1}/{len(eval_set)}] {str(row.get("label",""))[:35]}', end=' ')

            gt = str(row[target_field])

            paths   = get_image_paths(row)
            img_emb = None
            img_b64 = None
            for p in paths[:3]:
                img_emb = encode_image_from_disk(p, processor, dino, device)
                if img_emb is not None:
                    img_b64 = fetch_image_b64(p)
                    break
            if img_emb is None:
                img_emb = np.zeros(IMG_DIM, dtype=np.float32)

            text_emb = encode_text(row.to_dict(), text_model, blank_field=target_field)
            combined = np.concatenate([img_emb, text_emb]).astype(np.float32)
            norm = np.linalg.norm(combined)
            if norm > 0: combined = combined / norm

            # no self-filtering needed — eval set not in index
            _, I = index.search(combined.reshape(1,-1), TOP_K)
            neighbors = [
                id_map[int(idx)] for idx in I[0]
                if int(idx) in id_map and
                target_field in id_map[int(idx)].get('complete_fields',[])
            ][:TOP_K]

            prompt   = build_prompt(row.to_dict(), neighbors, target_field, vocab)
            messages = [{'role': 'user', 'content': []}]
            if img_b64:
                messages[0]['content'].append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{img_b64}', 'detail': 'low'}
                })
            messages[0]['content'].append({'type': 'text', 'text': prompt})

            try:
                response = client.chat.completions.create(
                    model='gpt-4o', messages=messages,
                    max_tokens=300, temperature=0.1,
                )
                raw    = response.choices[0].message.content.strip()
                raw    = raw.replace('```json','').replace('```','').strip()
                parsed = json.loads(raw)
                pred   = parsed.get(target_field)
                top3   = parsed.get(target_field + '_top3', [])
            except Exception as e:
                print(f'[GPT ERR: {e}]')
                pred, top3 = None, []

            metrics = compute_metrics(pred, gt, target_field, text_model)
            if target_field in CATEGORICAL_FIELDS and top3:
                metrics['top3_match'] = any(
                    str(c).strip().lower() == gt.strip().lower() for c in top3
                )

            for k, v in metrics.items():
                metric_accum.setdefault(k, []).append(float(v))

            print(f'GT={gt[:20]!r} PRED={str(pred)[:20]!r} '
                  f'EM={metrics["exact_match"]} '
                  f'Fuzz={metrics["fuzzy_token_sort"]:.2f} '
                  f'Sem={metrics["semantic_sim"]:.2f}')

            field_results.append({
                'uuid_hex':   str(row.get('uuid_hex','')),
                'label':      str(row.get('label','')),
                'item_class': str(row.get('item_class_label','')),
                'project':    str(row.get('project_label','')),
                'gt':         gt,
                'pred':       str(pred),
                'top3':       top3,
                **metrics,
            })

        agg = {k: round(float(np.mean(v)), 4) for k, v in metric_accum.items()}
        summary[target_field] = {'n': len(eval_set), **agg}
        all_results[target_field] = field_results

        print(f'\n  Results for {target_field}:')
        for k, v in agg.items():
            print(f'    {k:<25} {v:.3f}')

    print('\n' + '='*60)
    print('FINAL SUMMARY')
    print('='*60)
    for field, s in summary.items():
        print(f'\n  {field}')
        for k, v in s.items():
            if k != 'n':
                print(f'    {k:<25} {v:.3f}')

    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'summary': summary, 'results': all_results}, f, indent=2)
    print(f'\nWrote {OUTPUT_PATH}')

if __name__ == '__main__':
    main()