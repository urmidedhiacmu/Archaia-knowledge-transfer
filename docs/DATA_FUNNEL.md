# Data Funnel: 2.1M → 22,607

How we get from the full OpenContext database to the final v4 dataset.

## The three raw parquets

OpenContext exposes its data as three parquets:

| File | What it is | Size |
|---|---|---|
| `oc_all_manifest.parquet` | Entity registry — every artifact, site, project, media entity. ~2.1M rows. | ~1 GB |
| `oc_all_assertions.parquet` | Graph edges — links between entities with predicates. ~36.9M rows. | ~3 GB |
| `oc_all_resources.parquet` | Download URLs for media entities | ~500 MB |

**Critical data model note:** `resources.item_uuid` points to **media entities**,
not artifacts. Artifacts and media entities are both in the manifest under
`item_type = 'subjects'` and `item_type = 'media'` respectively.
A direct join of resources to artifacts returns nothing.

To get images for an artifact you need **3 hops**:
```
artifact.uuid
  → assertions[subject_uuid = artifact.uuid] → object_uuid  (= media entity UUID)
  → manifest[uuid = media UUID, item_type='media']
  → resources[item_uuid = media UUID] → uri               (download URL)
```

## The full funnel

| Count | Stage |
|---|---|
| 2,165,383 | Total subjects (item_type='subjects') in manifest |
| 207,062 | Media entities with at least 1 resource URL |
| 107,659 | Subjects with at least 1 downloadable image (verified via 3-hop join) |
| ~68,000 | After location + date filter |
| 33,412 | After image download and mapping reconstruction |
| 31,624 | After assertions recovery and augmentation (v1 parquet) |
| 22,607 | After removing non-artifact classes (v4 parquet — **final**) |

The biggest drop is the location + date filter: ~85% of image-bearing artifacts
have no coordinates or temporal data in OpenContext. These 55,831 artifacts
(those with images, valid item classes, but no spacetime) become the
**imputation targets** in `02_imputation/`.

## Dataset imbalances

- Heavily concentrated in the Middle East; minimal US coverage
- Largest site: ~200,000 artifacts; smallest sites: single digits
- 371,549:1 ratio between largest and smallest sites
- Different excavation projects use different standards — photo backgrounds,
  entry conventions, and collection methods vary significantly
- Models risk learning site-specific patterns rather than artifact features

## UUID handling

UUIDs appear in different encodings across the parquets:

| Context | Encoding | How to read |
|---|---|---|
| Raw parquets (`manifest`, `assertions`, `resources`) | 16-byte `bytes` | `uuid.UUID(bytes=b).hex` |
| v4 `item_class_uuid` column | Python bytes literal string e.g. `"b'\\x00...'"` | `ast.literal_eval(s).hex()` |
| v4 `uuid_hex` column | Plain hex string | Use directly |

Getting this wrong is a common source of silent join failures (0 rows matched).
