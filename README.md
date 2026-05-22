# ArchAIa — Knowledge Transfer Repository

Archaeological artifact dataset and field imputation pipeline built on the
[OpenContext](https://opencontext.org/) database, under Prof. Daphne Ippolito
at CMU's Language Technologies Institute.

**Artifact browser:** https://urmidedhiacmu.github.io/Archaia-viewer/
**Eval dashboard:** https://urmidedhiacmu.github.io/Archaia-viewer/eval/
**HuggingFace dataset (sample):** https://huggingface.co/datasets/archaia/dataset_sample_100_v4

---

## What was built

| Component | What it is |
|---|---|
| v4 dataset | 22,607 archaeological artifacts with images, coordinates, dates, and recovered metadata. Final parquet at `/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet` |
| Imputation pipeline | Retrieval-augmented field imputation for 55,831 more artifacts excluded from v4 for lacking spatial/temporal data |
| Artifact browser | GitHub Pages viewer for browsing artifacts with images and AI-generated use hypotheses |
| Eval dashboard | Per-field evaluation of the imputation pipeline (top-15 vs top-50 retrieval) |

---

## Repo structure

```
01_dataset_pipeline/    Raw OpenContext parquets → v4 dataset
02_imputation/          Field imputation pipeline (55,831 artifacts)
03_viewer/              Artifact browser build scripts
04_hf_upload/           HuggingFace dataset upload
docs/                   Design docs, bug write-ups, path reference
```

---

## Where to start

- **Understand the dataset:** read `docs/DATA_FUNNEL.md`, then run `01_dataset_pipeline/analysis/walkthrough.py`
- **Reproduce v4:** follow `01_dataset_pipeline/README.md`
- **Run imputation:** follow `02_imputation/README.md`
- **Rebuild the browser:** follow `03_viewer/README.md`

---

## Cluster

All compute runs on **Babel** (CMU HPC). Some scripts also reference the
**Schmidt Sciences** cluster. See `SETUP.md` for environment setup and
`docs/KEY_PATHS.md` for all important file paths.

---

## Key people

| Name | Role |
|---|---|
| Prof. Daphne Ippolito | PI, CMU LTI |
| Eric Kansa | OpenContext founder, external collaborator |
| Morris Alper | Team member |
| Mohith Rajesh | Team member |
| Eitan, Nicolas, Abdul | Team members |
| Shai Gordin | External advisor |
