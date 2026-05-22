# Imputation Pipeline Design

## Why imputation

v4 contains 22,607 artifacts that passed the full pipeline (location + date + images).
OpenContext has 55,831 more artifacts with images and valid item classes but no spatial
or temporal data — these were dropped at the spacetime filter.

The imputation pipeline attempts to fill in metadata fields for these 55,831 artifacts
using their images and item class as input, and v4 artifacts as a retrieval corpus.

## Model choices

### Embeddings

**Image: DINOv2 ViT-L/14** (`facebook/dinov2-large`)
- Self-supervised visual features, strong for object/material similarity
- 1024-dim output
- Load via HuggingFace, not torch.hub (Python 3.9 incompatibility)

**Text: GTE-Qwen2-7B-Instruct**
- Strong general-purpose text embeddings
- 384-dim output
- Encodes item_class_label + any available recovered fields

**Combined:** 1408-dim (1024 + 384), L2-normalized, FAISS IndexFlatIP (inner product = cosine on normalized vectors)

### Generation

**GPT-4o** (current)
- Receives: retrieved similar artifacts as structured JSON context + target artifact images + constrained vocabulary
- Produces: structured JSON with predicted field values
- API key: `ARCHAIA_OPENAI_API_KEY`

**Production plan: Qwen2.5-VL-72B**
- Multimodal, can process images directly
- Avoids OpenAI API costs
- Requires GPU with ~140GB VRAM (Schmidt Sciences cluster)

## Field selection rationale

Fields chosen for imputation must be:
1. Present in v4 at >30% fill rate (otherwise retrieval neighbors won't have them)
2. Cross-project generalizable (not project-specific vocabulary)
3. Inferrable from visual + item_class context

Fields excluded:
- `recovered_function` — 100% null, nothing to retrieve
- `recovered_chronotype` — Eric Kansa flagged that only 3 projects use it; their vocabulary isn't meaningful cross-project
- `recovered_fabric_group` — 94.7% missing, too sparse for retrieval
- `recovered_munsell_color` — physical measurement, can't infer from JPEG
- `recovered_size` — freeform text (e.g. "L: 4.3 cm"), not structured

## Retrieval design

**Why top-15 vs top-50?**
- Top-15: higher precision, less noise in context, faster/cheaper GPT-4o calls
- Top-50: more coverage of rare classes, but context window gets large and noisy

Eval compares both. Results in `eval_results/`.

**Train/eval split design**
Stratified by `(project_label, item_class_label)` to ensure both splits have
representation across all project × class combinations. This prevents the
model from being evaluated only on well-represented classes.
- Train: 19,215 artifacts (85%)
- Eval: 3,392 artifacts (15%)

## Known limitations

1. **Period imputation fails for bone-heavy sites** — the first 500 artifacts
   in `imputation_input.parquet` are mostly Human Bone from one project (Giza).
   These genuinely have no period signal in v4 neighbors, so `recovered_period`
   returns "Unknown". This is data sparsity, not a pipeline bug.

2. **Description repetitiveness for homogeneous classes** — GPT-4o defaults
   to a template when all retrieved neighbors look the same (e.g. undistinguished
   bone fragments). Descriptions become more varied for artifact-rich classes.

3. **Image loading at inference time** — imputation subjects' images are fetched
   on-the-fly from OpenContext URLs. This depends on network reliability and
   OpenContext uptime. If bulk download is preferable in the future, adapt
   `build_input.py` to store paths instead of URLs.

## Eval dashboard

Live at https://urmidedhiacmu.github.io/Archaia-viewer/eval/

Built as `eval/index.html` in the Archaia-viewer repo. Loads `top15.json`
and `top50.json`. Features:
- Per-field bar charts (exact match, fuzzy, semantic)
- Scatter plot (fuzzy vs semantic per artifact)
- Confusion matrix per categorical field
- Artifact browser with GT vs predicted side by side
