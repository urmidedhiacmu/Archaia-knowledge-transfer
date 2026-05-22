# Artifact Browser

The artifact browser is a GitHub Pages site that lets you browse the v4 dataset
with images, metadata, and AI-generated use hypotheses.

**Live:** https://urmidedhiacmu.github.io/Archaia-viewer/
**Repo:** https://github.com/urmidedhiaCMU/Archaia-viewer

The browser is a single `index.html` that loads `combined_data.json` — a
self-contained JSON with all artifact data and images embedded as base64.

---

## Files

| File | What it does |
|---|---|
| `build_combined_gh.py` | Builds `combined_data.json` from v4 + ablation results. Compresses and embeds images as base64. |
| `build_combined_gh.sh` | sbatch wrapper for the above (32GB, `general` partition) |
| `run_ablations_uses.py` | Generates AI use hypotheses for 20 artifacts of interest using GPT-4o and Gemini. Three prompt styles: `archaeological`, `museum_label`, `generative`. |
| `run_ablations_uses.sh` | sbatch wrapper for the above |

The `index.html` lives in the GitHub repo, not here.

---

## How to rebuild after dataset changes

1. Run ablations if you want updated use hypotheses (optional):
```bash
sbatch run_ablations_uses.sh
# outputs to ~/archaia_project/uses_outputs/ablation_results.json
```

2. Build the combined JSON on Babel:
```bash
sbatch build_combined_gh.sh
# outputs to ~/archaia_project/gh_deploy/combined_data.json
```

3. Copy to your local Archaia-viewer repo:
```bash
scp <andrewid>@login.babel.cs.cmu.edu:~/archaia_project/gh_deploy/combined_data.json \
    /path/to/Archaia-viewer/
```

4. Push to GitHub:
```bash
cd /path/to/Archaia-viewer
git add combined_data.json
git commit -m "update dataset"
git push origin main
```
GitHub Pages deploys automatically within ~2 minutes.

---

## `build_combined_gh.py` config

Key parameters at the top of the script:
```python
PARQUET       = '/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet'
IMAGE_DIR     = '/data/group_data/dei-group/archaia'
ABLATION_JSON = '~/archaia_project/uses_outputs/ablation_results.json'
SAMPLE_N      = 100       # artifacts in the browser
MAX_IMAGES    = all       # all deduplicated images per artifact
IMG_MAX_SIZE  = 800       # px, max dimension after resize
IMG_QUALITY   = 82        # JPEG quality
```

Output target is ~58 MB JSON (safe for GitHub Pages; limit is 100 MB).

---

## URL deep linking

Artifact URLs update as you browse:
`https://urmidedhiacmu.github.io/Archaia-viewer/?uuid=<uuid_hex>`

Sharing that link lands directly on that artifact. Implemented via
`history.pushState` in `index.html`.

---

## Eval dashboard

The eval dashboard lives at `/eval/` in the same repo.
It loads `eval/top15.json` and `eval/top50.json`.

To update after new eval runs:
```bash
cp ~/archaia_project/archaia_impute/outputs/eval_results_top15.json \
   /path/to/Archaia-viewer/eval/top15.json
cp ~/archaia_project/archaia_impute/outputs/eval_results_top50.json \
   /path/to/Archaia-viewer/eval/top50.json
git add eval/
git commit -m "update eval results"
git push origin main
```

---

## 20 artifacts of interest (for ablations)

Reconstructed Pottery Vessel, Bes plaque, Incense burner, Loom weight,
Stone weight, Coarse ware, Vessel neck, Figurine, Seal, Stamp seal,
Bichrome sherd, Painted lid, Bulla, Statuette, Lithic, Greenstone axe,
Ceramic lamp, Pestle, Tube, Stone vessel.

9 of these are v1-only (not in v4) — they were filtered out at the
spacetime stage. Their images must be fetched from OpenContext URLs directly.
