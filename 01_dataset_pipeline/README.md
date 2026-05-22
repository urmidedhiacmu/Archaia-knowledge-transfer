# Dataset Pipeline: Raw OpenContext → v4

This pipeline takes the three raw OpenContext parquets and produces
`archaia_final_dataset_augmented_v4.parquet` — 22,607 archaeological artifacts
with images, coordinates, dates, and recovered metadata fields.

For background on the data model, read `docs/DATA_FUNNEL.md` first.
To understand the code interactively, run `analysis/walkthrough.py`.

---

## Prerequisites

- Raw parquets downloaded to `~/archaia_project/data/`
- `archaia_env` activated
- ~200 GB free in `/data/user_data/<you>/archaia/`

---

## Pipeline steps — run in order

All scripts are in `pipeline/`. Steps 1–7 produce the base dataset.
Steps 8–13 fix bugs and augment it to produce v4.

### Step 1 — Download raw parquets
```bash
sbatch pipeline/01_download_data.sh
```
Downloads `oc_all_manifest.parquet`, `oc_all_assertions.parquet`,
`oc_all_resources.parquet` to `~/archaia_project/data/`.

---

### Step 2 — Filter artifacts
```bash
sbatch pipeline/02_filter_artifacts.sh
# or interactively:
python3 pipeline/02_filter_artifacts.py
```
Filters the manifest to subjects with location and date signal.
Outputs `artifacts_with_spacetime.csv`.

---

### Step 3 — Rank spacetime candidates
```bash
python3 pipeline/03_rank_spacetime_candidates.py
```
Scores and ranks artifacts by data quality (coordinate precision,
temporal depth, reference type). Outputs `artifacts_with_spacetime_ranked.csv`.

---

### Step 4 — Merge spacetime data
```bash
python3 pipeline/04_merge_spacetime_data.py
```
Joins the ranked candidates with the full spacetime parquet.

---

### Step 5 — Build image download list
```bash
python3 pipeline/05_build_image_download_list.py
```
Does the 3-hop join (artifact → assertions → media → resources → URL)
to find downloadable image URLs for each artifact.
Outputs `img2dataset_download_clean.tsv` to `/data/user_data/<you>/archaia/`.

> **Important:** This is a non-trivial join. Resources link to *media entities*,
> not directly to artifacts. A direct join of resources to artifacts returns
> nothing. See `docs/DATA_FUNNEL.md` for the full explanation.

---

### Step 6 — Download images
```bash
sbatch pipeline/06_download_images.sh
```
Runs `img2dataset` on the TSV produced in step 5.
Downloads ~387k images at 1024px to `/data/user_data/<you>/archaia/artifact_images_1024_full/`.
Takes 12–48 hours depending on network. Uses `--incremental True` so it's safe to resubmit.

> **Critical note on how img2dataset names files:**
> Output files are named by **TSV row index**, not by caption:
> row 0 → `000000000.jpg`, row 14169 → `000014169.jpg`.
> The caption (`artifact_<hex>_<seq>`) is only in a sidecar JSON.
> This naming scheme is the root cause of the image mapping bug — see `docs/BUGS_FIXED.md`.

---

### Step 7 — Build base dataset
```bash
sbatch pipeline/07_build_base_dataset.sh
# or:
python3 pipeline/07_build_base_dataset.py
```
Joins downloaded images with the spacetime-ranked artifact list.
Outputs `archaia_final_dataset.parquet` and `.csv` to `/data/user_data/<you>/archaia/final/`.

---

### Step 8 — Recover assertion text fields (→ v1)
```bash
python3 pipeline/08_recover_assertion_text_fields.py
```
Extracts human-readable metadata from the assertions parquet using exact
predicate UUID lookups. Fields recovered: `recovered_material`,
`recovered_object_type`, `recovered_condition`, `recovered_period`,
`recovered_description`, `recovered_note`, and others.

> **Critical implementation note:** field values must be extracted using
> **exact predicate hex UUIDs**, not label string matching.
> Values come from either `obj_string` or `obj_uuid` (resolved via manifest
> entity label map) depending on the field. Getting this wrong produces 0% fill rates.
> See `docs/BUGS_FIXED.md` for the predicate UUID table.

Outputs `archaia_final_dataset_augmented_text_v1.parquet` to
`/data/group_data/dei-group/archaia/`.

---

### Step 9 — Rebuild image paths (v1 → v2)
```bash
python3 pipeline/09_rebuild_image_paths.py
```
**This fixes the critical image mapping bug** — in v1, every artifact had
the wrong images due to an incorrect offset in the original reconstruction.

The fix scans every image on disk, reads the filename number as a TSV row
index, looks up the artifact hex from the TSV caption column, and assigns
images to the correct artifact. All 31,624 rows are corrected.

Outputs v2 to `/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v2.parquet`.

See `docs/BUGS_FIXED.md` for full details.

---

### Step 10 — Fix uuid_hex and merge (v2 → v3)
```bash
python3 pipeline/10_fix_uuid_and_merge.py
```
Fixes the missing `uuid_hex` column (dropped during assertions recovery),
re-joins via the `slug` column, and reconciles the 1,788-row discrepancy
between the base CSV and the v1 parquet.

Outputs v3.

---

### Step 11 — Drop temporary columns
```bash
python3 pipeline/11_drop_temp_columns.py
```
Cleans up intermediate columns added during the fix process.
Overwrites v3 in place.

---

### Step 12 — Filter non-artifact classes (v3 → v4)
```bash
python3 pipeline/12_filter_nonartifact_classes.py
```
Removes spatial recording units (Loci, Trenches, Survey Units, Sites,
Contexts, Units) that passed the spacetime filter because loci have
coordinates. Reduces from 31,624 → 22,607 rows.

See `docs/BUGS_FIXED.md` for the full list of removed classes and counts.

Outputs v4 to `/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet`.

---

### Step 13 — Deduplicate image paths (overwrites v4)
```bash
python3 pipeline/13_deduplicate_image_paths.py
```
OpenContext stores the same physical image under 3 URLs (archive.org original,
GCS preview, GCS thumbnail). This collapses them by URL filename stem.
Two-pass dedup: 102,140 → 78,114 image references.

Overwrites v4 in place.

---

## Final output

`/data/group_data/dei-group/archaia/archaia_final_dataset_augmented_v4.parquet`

| Field | Value |
|---|---|
| Rows | 22,607 |
| Columns | 53 |
| Image references | 78,114 |
| Mean images/artifact | ~3.5 |
| Lat/lon coverage | 100% |
| Temporal coverage | 94.6% |

---

## Analysis scripts

Not part of the pipeline — run these to explore or verify:

| Script | What it does |
|---|---|
| `analysis/walkthrough.py` | **Start here.** Full annotated walkthrough of the data model, the 3-hop image join, the bug, and the fix. Written for code review. |
| `analysis/explore_data.py` | EDA on the raw parquets |
| `analysis/explore_ranked_dataset.py` | EDA on the spacetime-ranked dataset |
| `analysis/analyze_dataset.py` | Column completeness analysis |
| `analysis/availability.py` | Field availability across projects |
| `analysis/sanity_check_useful.py` | Spot checks on v4 |
| `analysis/inspect_new_parquet.py` | Quick parquet inspector |
| `analysis/preview_recovered_subset.py` | Preview recovered assertion fields |
| `analysis/check_item_class_distribution.py` | Item class breakdown (used to decide what to remove in step 12) |
| `analysis/check_artifact_counts.py` | Artifact counts by project/class |
| `analysis/photo_classification.py` | Image content analysis |
