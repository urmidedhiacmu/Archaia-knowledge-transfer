# Archaia Dataset Columns

## Overview

This is the cleaned artifact-level dataset with recovered text metadata added from the OpenContext assertion layer.

- Rows: **31,624**
- Columns: **51**

The original dataset already contained the artifact rows, project/source information, spatial fields, temporal fields, and image paths.
The recovered fields were added later by scanning OpenContext assertions and pulling back descriptive text fields that were missing.

## High-level pipeline

1. Start from the cleaned final artifact dataset (`archaia_final_dataset`).
2. Use artifact UUIDs from that dataset to scan the OpenContext assertions parquet.
3. Resolve assertion values using manifest lookups:
   - use object labels when the value is stored as a linked object
   - use primitive assertion fields (`obj_string`, `obj_datetime`, etc.) when the object is just a null placeholder
4. Keep useful descriptive predicates such as material, note, description, object type, size, period, condition, etc.
5. Add selected recovered values back as separate `recovered_*` columns.
6. Store the full recovered assertion text for each artifact in `recovered_text_fields_json`.

## Recovered text fields

The `recovered_*` columns are individual fields pulled out from assertions, for example recovered material, note, description, object type, and condition.
Coverage differs by field because different source projects used different schemas and recorded different kinds of metadata.

### `recovered_text_fields_json`

This is the full recovered text metadata for an artifact stored as a JSON object.
It keeps the original assertion field names as keys and stores values as lists, since an artifact can have more than one value for the same field.

Example shape:

```json
{
  "Material": ["Obsidian"],
  "Artifact Name": ["Mirror"],
  "Has note": ["Fragment of a very well polished obsidian piece..."]
}
```

Use this column when you want the most complete recovered metadata without losing field names.

## Column table

| Column | Type | Coverage | Description |
|---|---|---:|---|
| `label` | `object` | 100.0% | Artifact label or catalog identifier used by the excavation project. |
| `metadata` | `object` | 4.6% | Additional metadata from the source project. |
| `project_label` | `object` | 100.0% | Name of the archaeological project. |
| `slug` | `object` | 100.0% | URL-friendly identifier used by OpenContext for the artifact. |
| `is_best` | `bool` | 100.0% | Whether this record was selected as the best spatial/temporal reference. |
| `quality_score` | `float64` | 100.0% | Score used when selecting the best reference record. |
| `reference_type` | `object` | 100.0% | Type of reference used for location or chronology. |
| `context_uuid` | `object` | 100.0% | UUID of the archaeological context. |
| `item_class_uuid` | `object` | 100.0% | UUID identifying the artifact class. |
| `project_uuid` | `object` | 100.0% | UUID of the project in OpenContext. |
| `geo_depth` | `float64` | 100.0% | Depth of spatial inheritance used when assigning location. |
| `geo_note` | `float64` | 0.0% | Notes related to geographic assignment. |
| `geo_specificity_y` | `float64` | 100.0% | Specificity ranking of the geographic reference. |
| `geo_zoom` | `object` | 0.07% | Zoom level used when deriving geographic coordinates. |
| `geometry` | `object` | 100.0% | GeoJSON geometry describing artifact location. |
| `geometry_type` | `object` | 100.0% | Geometry type (typically Point). |
| `latitude` | `float64` | 100.0% | Latitude of artifact location. |
| `longitude` | `float64` | 100.0% | Longitude of artifact location. |
| `image_count_y` | `int64` | 100.0% | Number of images linked to the artifact. |
| `image_paths` | `object` | 100.0% | Local file paths of downloaded artifact images. |
| `chrono_depth` | `float64` | 85.99% | Depth of chronological inheritance. |
| `earliest` | `float64` | 85.99% | Earliest possible date associated with artifact. |
| `latest` | `float64` | 85.99% | Latest possible date associated with artifact. |
| `start` | `float64` | 85.99% | Start date of artifact's temporal range. |
| `stop` | `float64` | 85.99% | End date of artifact's temporal range. |
| `recovered_artifact_name` | `object` | 2.67% | Artifact name recovered from OpenContext assertions. |
| `recovered_material` | `object` | 29.47% | Material of the artifact recovered from assertions. |
| `recovered_material_note` | `object` | 3.92% | Additional notes about artifact material. |
| `recovered_note` | `object` | 41.27% | Recovered general note text. |
| `recovered_description` | `object` | 38.42% | Recovered main description text. |
| `recovered_description_remarks` | `object` | 2.66% | Recovered extra remarks related to the description. |
| `recovered_object_type` | `object` | 20.96% | Recovered object type or category. |
| `recovered_object_type_note` | `object` | 17.39% | Recovered note about object type. |
| `recovered_period` | `object` | 5.51% | Recovered named archaeological period. |
| `recovered_chronotype` | `object` | 11.39% | Recovered chronotype classification used by the source project. |
| `recovered_fabric_description` | `object` | 17.06% | Recovered description of ceramic or material fabric. |
| `recovered_fabric_group` | `object` | 3.79% | Recovered fabric group classification. |
| `recovered_munsell_color` | `object` | 11.24% | Recovered Munsell color text. |
| `recovered_munsell_number` | `object` | 7.14% | Recovered Munsell numeric/code value. |
| `recovered_decorative_technique` | `object` | 6.48% | Recovered decorative technique field. |
| `recovered_size` | `object` | 19.2% | Recovered size or measurement text. |
| `recovered_specific_context` | `object` | 0.97% | Recovered specific archaeological context. |
| `recovered_specific_location` | `object` | 1.08% | Recovered specific location within the site. |
| `recovered_location` | `object` | 7.58% | Recovered general location description. |
| `recovered_locus` | `object` | 5.57% | Recovered excavation locus value. |
| `recovered_locus_id` | `object` | 11.28% | Recovered locus identifier. |
| `recovered_function` | `object` | 10.7% | Recovered interpreted function. |
| `recovered_condition` | `object` | 23.8% | Recovered preservation or condition text. |
| `recovered_registration_date` | `object` | 2.67% | Recovered registration or cataloguing date. |
| `recovered_disposition` | `object` | 1.49% | Recovered storage, museum, or disposition field. |
| `recovered_text_fields_json` | `object` | 99.41% | JSON object containing all recovered assertion text fields for the artifact. |

## Original columns kept

- `label`
- `metadata`
- `project_label`
- `slug`
- `is_best`
- `quality_score`
- `reference_type`
- `context_uuid`
- `item_class_uuid`
- `project_uuid`
- `geo_depth`
- `geo_note`
- `geo_specificity_y`
- `geo_zoom`
- `geometry`
- `geometry_type`
- `latitude`
- `longitude`
- `image_count_y`
- `image_paths`
- `chrono_depth`
- `earliest`
- `latest`
- `start`
- `stop`

## Recovered columns added

- `recovered_artifact_name`
- `recovered_material`
- `recovered_material_note`
- `recovered_note`
- `recovered_description`
- `recovered_description_remarks`
- `recovered_object_type`
- `recovered_object_type_note`
- `recovered_period`
- `recovered_chronotype`
- `recovered_fabric_description`
- `recovered_fabric_group`
- `recovered_munsell_color`
- `recovered_munsell_number`
- `recovered_decorative_technique`
- `recovered_size`
- `recovered_specific_context`
- `recovered_specific_location`
- `recovered_location`
- `recovered_locus`
- `recovered_locus_id`
- `recovered_function`
- `recovered_condition`
- `recovered_registration_date`
- `recovered_disposition`
- `recovered_text_fields_json`