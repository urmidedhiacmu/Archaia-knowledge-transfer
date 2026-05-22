# Imputation Pipeline

Retrieval-augmented field imputation for 55,831 artifacts that have images
and valid item classes in OpenContext but were excluded from v4 for lacking
spatial or temporal data (dropped at the spacetime filter stage).

v4 (22,607 artifacts) is the ground truth. A FAISS index is built over v4's
train split. For each imputation subject, visually and textually similar
artifacts are retrieved, and GPT-4o generates structured field predictions
using those neighbors as context.

---

## Target fields

| Field | Fill rate in v4 | Notes |
|---|---|---|
| `recovered_material` | ~65% | e.g. "Bone", "Clay", "Bronze" |
| `recovered_object_type` | ~58% | e.g. "Vessel", "Coin", "Figurine" |
| `recovered_condition` | ~40% | e.g. "Fragmentary", "Complete" |
| `recovered_period` | ~52% | e.g. "Bronze Age", "Roman" |
| `recovered_description` | ~35% | free text |

Fields explicitly excluded and why:
- `recovered_function` — 100% null in v4, nothing to retrieve from
- `recovered_chronotype` — only 3 projects use it, not cross-project generalizable (Eric Kansa flagged this)
- `recovered_fabric_group` — 94.7% missing
- `recovered_munsell_color` — requires physical examination, can't infer from images
- `recovered_size` — freeform measurement text, not structured

---

## Architecture

```
imputation_input.parquet  (55,831 artifacts, image URLs, item_class, partial fields)
        |
        v
   encode.py  ──  DINOv2 ViT-L/14 (1024-dim image embeddings)
                + GTE-Qwen2-7B-Instruct (384-dim text embeddings)
                = 1408-dim L2-normalized vectors
        |
        v
   FAISS IndexFlatIP  (built over v4 TRAIN split: 19,215 artifacts)
        |
        v
   retrieve.py  ──  top-k nearest neighbors
        |
        v
   generate.py  ──  GPT-4o with retrieved artifacts as JSON context
                    + constrained vocabulary per field
        |
        v
   imputed.parquet  (55,831 rows with predicted fields)
```

---

## Important implementation notes

**DINOv2 loading:** Load via HuggingFace (`facebook/dinov2-large`), NOT `torch.hub`.
Torch hub uses Python `X | Y` union type syntax which breaks on Python 3.9 (Babel's version).

```python
# Correct:
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-large")

# Broken on Python 3.9:
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
```

**API key:** stored in `~/archaia_project/scripts/.env` as `ARCHAIA_OPENAI_API_KEY`.
The sbatch wrappers source this automatically.

**Image loading:** imputation subjects never had their images downloaded to disk.
Images are fetched on-the-fly from `image_urls` (top 3 OpenContext URLs stored per
artifact in `imputation_input.parquet`). No bulk download needed.

**v4 train/eval split:** 85/15 stratified by `(project_label, item_class_label)`.
FAISS index is built over the **train split only** (19,215 artifacts).
Eval set (3,392 artifacts) is used for evaluation — never seen during indexing.

---

## How to run

Activate environment and set up API key first (see `SETUP.md`).

Run the four steps in order:

```bash
# Step 1: Build imputation input dataset (~4 hours, needs 64GB)
sbatch scripts/00_build_input.sh

# Step 2: Encode v4 + build FAISS index (~6 hours, needs GPU)
sbatch scripts/01_build_index.sh

# Step 3: Run imputation on all 55,831 artifacts (~6 hours, needs GPU + OpenAI key)
sbatch scripts/02_impute.sh

# Step 4: Evaluate on held-out v4 eval split (~8 hours, needs GPU + OpenAI key)
sbatch scripts/03_eval.sh
```

Monitor jobs:
```bash
squeue -u $USER
tail -f archaia_impute/logs/build_input_<jobid>.out
```

---

## Outputs

| File | What it is |
|---|---|
| `archaia_impute/data/imputation_input.parquet` | 55,831 imputation subjects |
| `archaia_impute/index/` | FAISS index files |
| `archaia_impute/outputs/imputed.parquet` | Full imputation results |
| `eval_results/eval_results_top15.json` | Eval scores, top-15 retrieval |
| `eval_results/eval_results_top50.json` | Eval scores, top-50 retrieval |

---

## Evaluation

Eval is run per-field independently: 100 artifacts sampled from the held-out
v4 eval split, target field blanked, pipeline run, prediction scored against
ground truth.

Two configs compared: top-15 vs top-50 neighbors passed to GPT-4o.

Metrics:
- **Exact match** — all fields
- **Fuzzy token sort ratio** — all fields
- **Semantic similarity** (sentence embedding cosine) — all fields
- **Top-3 accuracy** (GT appears in model's top 3 candidates) — categorical fields
- **BLEU** — `recovered_description` only

Results are visualized at: https://urmidedhiacmu.github.io/Archaia-viewer/eval/

To rebuild the eval dashboard after new eval runs, see `03_viewer/README.md`.

---

## 00_data_prep scripts

| Script | What it does |
|---|---|
| `build_input.py` | Main script. Builds `imputation_input.parquet` from the 3 raw OC parquets. Does the 3-hop image join, removes v4 artifacts, extracts partial recovered fields. |
| `audit.py` | Missingness stats → `audit_report.json`. Run this to understand which fields are worth imputing. |
| `split.py` | Produces the 85/15 stratified train/eval split of v4 |
| `entropy.py` | Field entropy analysis |
| `vocab.py` | Builds constrained vocabulary per field from v4 train split. Used by `generate.py` to constrain GPT-4o output. |
