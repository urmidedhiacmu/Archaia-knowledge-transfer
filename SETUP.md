# Setup

## Cluster access

All scripts run on **Babel** (CMU HPC).
Login: `ssh <andrewid>@login.babel.cs.cmu.edu`

Interactive debug session (use this for testing before submitting jobs):
```bash
srun --partition=debug --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash
```

For GPU jobs use `--partition=general --gres=gpu:1 --qos=normal`.
The `normal` QoS **requires at least 1 GPU** — CPU-only jobs must use `--partition=cpu`.

---

## Python environment

A virtualenv already exists at `/home/udedhia/archaia_env`.
If you're a new user, create your own:

```bash
python3 -m venv ~/archaia_env
source ~/archaia_env/bin/activate
pip install pandas pyarrow faiss-cpu sentence-transformers torch torchvision \
    transformers openai requests pillow rapidfuzz nltk huggingface_hub \
    img2dataset --break-system-packages
```

Activate before running anything:
```bash
source ~/archaia_env/bin/activate
```

---

## API keys

The imputation pipeline (`02_impute/` and `03_eval/`) uses OpenAI GPT-4o.
The key is stored in a `.env` file that is **not committed to this repo**.

Create it at `~/archaia_project/scripts/.env`:
```bash
ARCHAIA_OPENAI_API_KEY=sk-...
```

The sbatch wrappers source this file automatically:
```bash
source /home/<you>/archaia_project/scripts/.env
export ARCHAIA_OPENAI_API_KEY
```

---

## Storage layout on Babel

| Path | What's there | Size |
|---|---|---|
| `~/archaia_project/` | All scripts and code | — |
| `~/archaia_project/data/` | Raw OpenContext parquets | ~4 GB |
| `/data/user_data/<you>/archaia/` | Downloaded images + TSV | ~200 GB |
| `/data/user_data/<you>/archaia/final/images/` | Symlinked/canonical image dir | — |
| `/data/group_data/dei-group/archaia/` | Shared datasets (v1–v4 parquets, ablation results) | ~50 GB |

See `docs/KEY_PATHS.md` for the full path reference.

---

## Raw OpenContext parquets

Downloaded from Google Cloud Storage:
```
https://storage.googleapis.com/opencontext-parquet/oc_all_manifest.parquet
https://storage.googleapis.com/opencontext-parquet/oc_all_resources.parquet
https://storage.googleapis.com/opencontext-parquet/oc_all_assertions.parquet
```
There is also `oc_all_manifest_cached_spacetime.parquet` — contact Eric Kansa
(eric@opencontext.org) for access as it may not be publicly posted.

---

## Useful columns reference

`archaia_useful_columns_availability.csv` and `archaia_useful_columns_sample.csv`
in `01_dataset_pipeline/analysis/` document which metadata columns exist and
their fill rates across the dataset.
