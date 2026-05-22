# Bugs Found and Fixed

These bugs were discovered during the dataset audit and fixed before v4.
A future student re-running the pipeline from scratch will not hit them
because the pipeline scripts already contain the fixes. This document
explains what went wrong in case you encounter related issues.

---

## Bug 1 (Critical): Image paths wrong for all 31,624 artifacts

**Severity:** Critical — every artifact in v1 had images from a different artifact.

### How img2dataset names output files

img2dataset input is a TSV with columns `url` and `caption`:
```
url                                    caption
https://archive.org/.../img1.jpg       artifact_abc123_001
https://archive.org/.../img2.jpg       artifact_abc123_002
https://archive.org/.../img3.jpg       artifact_def456_001
```

Output files are named by **TSV row index**, not by caption:
- row 0 → `000000000.jpg`
- row 14169 → `000014169.jpg`

The caption is stored in a sidecar metadata JSON. The filename number is the
only link back to the TSV row (and thus back to the artifact).

### What went wrong

The original reconstruction that mapped img2dataset output files back to
artifacts used an incorrect offset/sort order. The result was a systematic
shift across all 31,624 rows — every artifact had images from a different
artifact.

Verification: take any artifact's first image filename → parse the row number
→ look up that TSV row → the caption should contain the artifact's UUID.
10/10 spot checks failed before the fix.

### The fix (`09_rebuild_image_paths.py`)

Rebuilt image_paths entirely from scratch:
1. Scan every image on disk
2. Parse filename number → TSV row index
3. Read caption at that row → extract artifact hex
4. Assign image to that artifact
5. Join back to dataset on `uuid_hex`

Post-fix: 10/10 random spot checks passed.

---

## Bug 2: `uuid_hex` column missing from v1

The `uuid_hex` column present in the base CSV was dropped during the
assertions recovery step, making it impossible to join v1 back to the base
CSV or the mapping CSV.

**Fix:** Re-joined v1 to the base CSV via the `slug` column (100% populated
in both files, zero duplicates). All 31,624 rows matched. `uuid_hex` added
back permanently. (`10_fix_uuid_and_merge.py`)

---

## Bug 3: 1,788 row discrepancy between base CSV and v1

The base CSV had 33,412 rows while v1 had 31,624. The 1,788 rows were lost
during the assertions recovery step due to a merge that dropped artifacts
with no assertion matches.

**Fix:** The row count difference is real — those 1,788 artifacts genuinely
had no recovered assertion fields. The slug-based re-join in bug 2's fix
correctly handles this.

---

## Bug 4: Non-artifact classes in v1

`manifest[item_type='subjects']` includes both physical artifacts and
excavation recording units (Loci, Trenches, Survey Units, Sites, Contexts,
Units). These all passed the spacetime filter because loci have coordinates.

**Classes removed in v4** (`12_filter_nonartifact_classes.py`):

| Class | Count removed | Reason |
|---|---|---|
| Locus | 6,947 | Spatial subdivision of excavation |
| Survey Unit | 1,699 | Geographic zone for surface survey |
| Unit | 179 | Stratigraphic excavation unit |
| Site | 134 | Entire site record |
| Trench | 50 | Excavation trench |
| Context | 8 | Stratigraphic context |
| **Total** | **9,017** | 31,624 → 22,607 |

`Feature` and `Structure` were **kept** — Feature includes real portable
artifacts (grinding stones, etc.) alongside stratigraphic deposits and they
can't be separated programmatically.

---

## Bug 5: Image duplicates (3x per image)

OpenContext stores the same physical image under three different URLs:
- `https://archive.org/download/.../img.jpg` (original)
- `https://storage.googleapis.com/.../ia-previews/img.jpg` (GCS preview)
- `https://storage.googleapis.com/.../ia-thumbnails/img.jpg` (GCS thumbnail)

All three appear as separate TSV rows → three downloaded files → three entries
in `image_paths` for the same physical image.

**Fix (`13_deduplicate_image_paths.py`):** Two-pass dedup:
1. Deduplicate on (artifact_hex, URL) pairs → removes exact URL duplicates
2. Deduplicate on URL filename stem → collapses archive.org + GCS triples

Result: 102,140 → 78,114 image references.

---

## Assertions recovery: predicate UUIDs

When extracting recovered fields from the assertions parquet, you **must**
use exact predicate hex UUIDs. Label string matching returns nothing.

| Field | Predicate hex UUID | Value source |
|---|---|---|
| `recovered_material` | `de0970679ad05d48fb02e1905c46fefa` | `obj_uuid` → resolve to entity label |
| `recovered_object_type` | `7db79382743242a4fbc5ef760691905a` | `obj_uuid` → resolve to entity label |
| `recovered_condition` | `4909306f310247a266a3561c296147bb` | `obj_string` directly |
| `recovered_period` | `0b643ab938a44f450e41415c45cb7702` | `obj_uuid` → resolve to entity label |
| `recovered_chronotype` | `13d9229565ea47f7ebf256c7667c6e5f` | `obj_uuid` → resolve to entity label |
| `recovered_decorative_technique` | `f07c30bc6c714c977893d61ff6d0b59b` | `obj_uuid` → resolve to entity label |
| `recovered_fabric_group` | `423ba1ec3cd44dba40eb9474c1ae0d3a` | `obj_uuid` → resolve to entity label |
| `recovered_description` | `7dbb5cb7599f42d561ee1955cf898990` | `obj_string` directly |
| `recovered_note` | `5fa8fc7574f8725a489db727c033d79c` | `obj_string` directly |

`obj_uuid` fields: the value is an entity UUID in the manifest — look it up
in the manifest `label` column to get the human-readable string.
`obj_string` fields: the value is already a string in `obj_string`.
