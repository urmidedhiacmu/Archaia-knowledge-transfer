# encode.py — now uses v4_train.parquet instead of full v4
# Everything else identical

import os, json, ast
import numpy as np
import pandas as pd
from PIL import Image

TRAIN_PATH   = '/home/udedhia/archaia_project/archaia_impute/data/v4_train.parquet'
IMAGE_ROOT   = '/data/user_data/udedhia/archaia/final'
INDEX_DIR    = '/home/udedhia/archaia_project/archaia_impute/index'

IMPUTE_FIELDS = [
    'recovered_material', 'recovered_object_type', 'recovered_condition',
    'recovered_period', 'recovered_description'
]

def get_image_paths(row):
    raw = row.get('image_paths', [])
    if isinstance(raw, str):
        try: return json.loads(raw)
        except:
            try: return ast.literal_eval(raw)
            except: return []
    if hasattr(raw, 'tolist'): return raw.tolist()
    return list(raw) if raw is not None else []

def encode_text(row, text_model):
    parts = []
    for f in IMPUTE_FIELDS:
        v = row.get(f)
        if v and str(v) not in ('nan','None',''):
            parts.append(f"{f.replace('recovered_','')}: {str(v)[:200]}")
    parts.append(f"class: {row.get('item_class_label','')}")
    parts.append(f"project: {row.get('project_label','')}")
    text = ' | '.join(parts) if parts else str(row.get('label',''))
    return text_model.encode([text], show_progress_bar=False)[0]

def main():
    import torch, faiss
    from transformers import AutoImageProcessor, AutoModel
    from sentence_transformers import SentenceTransformer

    os.makedirs(INDEX_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Loading DINOv2...')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    dino      = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    dino.eval()

    print('Loading MiniLM...')
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print('Loading train split...')
    df = pd.read_parquet(TRAIN_PATH)
    print(f'  {len(df):,} artifacts')

    IMG_DIM  = 1024
    TEXT_DIM = 384
    TOTAL    = IMG_DIM + TEXT_DIM

    vectors = []
    id_map  = {}

    print('Encoding...')
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f'  [{i}/{len(df)}]')

        paths   = get_image_paths(row)
        img_emb = None
        for p in paths[:3]:
            try:
                full   = os.path.join(IMAGE_ROOT, p)
                img    = Image.open(full).convert('RGB')
                inputs = processor(images=img, return_tensors='pt').to(device)
                with torch.no_grad():
                    out = dino(**inputs)
                img_emb = out.last_hidden_state[:,0,:].cpu().numpy().flatten()
                break
            except:
                continue
        if img_emb is None:
            img_emb = np.zeros(IMG_DIM, dtype=np.float32)

        text_emb = encode_text(row.to_dict(), text_model)
        combined = np.concatenate([img_emb, text_emb]).astype(np.float32)
        norm     = np.linalg.norm(combined)
        if norm > 0: combined = combined / norm

        complete_fields = [
            f for f in IMPUTE_FIELDS
            if pd.notna(row.get(f)) and str(row.get(f,'')) not in ('nan','None','')
        ]

        id_map[len(vectors)] = {
            'row_index':       int(idx),
            'uuid_hex':        str(row.get('uuid_hex','')),
            'label':           str(row.get('label','')),
            'item_class':      str(row.get('item_class_label','')),
            'project':         str(row.get('project_label','')),
            'complete_fields': complete_fields,
            'recovered_material':    str(row.get('recovered_material','')),
            'recovered_object_type': str(row.get('recovered_object_type','')),
            'recovered_condition':   str(row.get('recovered_condition','')),
            'recovered_period':      str(row.get('recovered_period','')),
            'recovered_description': str(row.get('recovered_description',''))[:300],
        }
        vectors.append(combined)

    print('Building FAISS index...')
    matrix = np.stack(vectors).astype(np.float32)
    index  = faiss.IndexFlatIP(TOTAL)
    index.add(matrix)
    faiss.write_index(index, os.path.join(INDEX_DIR, 'faiss.index'))
    with open(os.path.join(INDEX_DIR, 'id_map.json'), 'w') as f:
        json.dump(id_map, f)
    print(f'Done. {index.ntotal:,} vectors, dim={TOTAL}')

if __name__ == '__main__':
    main()