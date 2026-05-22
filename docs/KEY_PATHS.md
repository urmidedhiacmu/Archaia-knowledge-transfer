# Key Paths on Babel

## Code

| Path | What's there |
|---|---|
| `~/archaia_project/` | All scripts |
| `~/archaia_project/data/` | Raw OpenContext parquets |
| `~/archaia_project/archaia_impute/` | Imputation pipeline |
| `~/archaia_project/gh_deploy/` | Browser build outputs (`combined_data.json`) |
| `~/archaia_project/uses_outputs/` | Ablation results (`ablation_results.json`) |
| `~/archaia_project/scripts/.env` | API keys — **not in git** |

## Data (large files)

| Path | What's there |
|---|---|
| `/data/user_data/<you>/archaia/` | Your personal data dir |
| `/data/user_data/<you>/archaia/artifact_images_1024_full/` | Downloaded images (~186k files) |
| `/data/user_data/<you>/archaia/img2dataset_download_clean.tsv` | Image URL list (387,182 rows, 33,412 artifacts) |
| `/data/user_data/<you>/archaia/final/` | Base dataset outputs |
| `/data/user_data/<you>/archaia/final/images/` | Canonical image dir (symlinked to group) |
| `/data/group_data/dei-group/archaia/` | **Shared group dir** — all parquet versions |
| `/data/group_data/dei-group/archaia/images/` | Images symlinked from user dir (for team access) |

## Dataset versions (in `/data/group_data/dei-group/archaia/`)

| File | What it is |
|---|---|
| `archaia_final_dataset.parquet` / `.csv` | Base dataset (pre-augmentation) |
| `archaia_final_dataset_augmented_text_v1.parquet` | After assertions recovery |
| `archaia_final_dataset_augmented_v2.parquet` | After image path fix |
| `archaia_final_dataset_augmented_v3.parquet` | After uuid_hex fix + cleanup |
| `archaia_final_dataset_augmented_v4.parquet` | **Final** — after item_class filter + image dedup |

**Use v4 for everything.** Earlier versions are kept for audit purposes only.

## Imputation pipeline data

| Path | What's there |
|---|---|
| `~/archaia_project/archaia_impute/data/imputation_input.parquet` | 55,831 imputation subjects |
| `~/archaia_project/archaia_impute/index/` | FAISS index files |
| `~/archaia_project/archaia_impute/outputs/imputed.parquet` | Full imputation predictions |
| `~/archaia_project/archaia_impute/outputs/eval_results_top15.json` | Eval scores, top-15 |
| `~/archaia_project/archaia_impute/outputs/eval_results_top50.json` | Eval scores, top-50 |

## Python environment

```bash
source ~/archaia_env/bin/activate
```

## Giving a teammate access to images

```bash
# Symlink your images dir into the group dir so others can read it
ln -s /data/user_data/<you>/archaia/final/images \
      /data/group_data/dei-group/archaia/images
```
