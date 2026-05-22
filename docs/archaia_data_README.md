# Archaia — Hierarchical Archaeological Artifact Descriptions Dataset

A multimodal dataset of **31,600+** excavated archaeological artifacts, each with multi-angle photographs, structured excavation metadata, and **5-level AI-generated hierarchical descriptions** ranging from a single-word category to a full publication-ready report.

---

## Files

```
/data/group_data/dei-group/archaia/
│
├── archaia_hf_final/                         # HuggingFace Dataset (load_from_disk)
├── archaia_hf_final.csv                      # Same data as CSV (no images)
│
├── archaia_final_dataset.csv                 # Source CSV with all artifact metadata
├── archaia_final_dataset_descriptions.jsonl  # Raw generation output (JSONL)
├── archaia_final_dataset_descriptions_clean.jsonl  # Cleaned version (errors removed)
│
├── images/                                   # Artifact photographs on disk
│   ├── 00001/
│   │   ├── 000016965.jpg
│   │   └── ...
│   └── ...
│
└── README.md                                 # This file
```

---

## Loading the Dataset

### HuggingFace Dataset

```python
from datasets import load_from_disk

ds = load_from_disk("/data/group_data/dei-group/archaia/archaia_hf_final")
print(ds)
# Dataset({
#     features: ['artifact_id', 'description', 'level_1_description', ..., 'image_paths', 'metadata', ...],
#     num_rows: 31605
# })

# Access a single row
row = ds[0]
print(row["artifact_id"])
print(row["level_5_description"])
```

### CSV

```python
import pandas as pd

df = pd.read_csv("/data/group_data/dei-group/archaia/archaia_hf_final.csv")
print(df.shape)          # (31605, N)
print(df.columns.tolist())
```

### Loading Images from Paths

The dataset stores **relative image paths** as a JSON-encoded list. To load the actual images:

```python
import json
from pathlib import Path
from PIL import Image

IMAGES_ROOT = Path("/data/group_data/dei-group/archaia")

row = ds[0]
rel_paths = json.loads(row["image_paths"])   # e.g. ["images/00001/000016965.jpg", ...]

images = []
for p in rel_paths:
    img = Image.open(IMAGES_ROOT / p).convert("RGB")
    images.append(img)

# Display
images[0].show()
```

---

## Column Reference

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `artifact_id` | string | Unique artifact identifier, e.g. `artifact_0911ea67171e4a1d94af41664f5b8085` |
| 2 | `description` | string | Original catalog description written by archaeologists at excavation time |
| 3 | `level_1_description` | string | **Object class** — broadest category, e.g. *"Cover Tile Fragment"*, *"Vessel"*, *"Coin"* |
| 4 | `level_2_description` | string | **Material + sub-type** — adds material and refined type, e.g. *"Terracotta cover tile fragment"* |
| 5 | `level_3_description` | string | **Visual identity** — prose description of form, color, condition from images + metadata |
| 6 | `level_4_description` | string | **Full analytical** — all metadata fields: Munsell color, dimensions, decorative techniques, trench coordinates, surface observations |
| 7 | `level_5_description` | string | **Publication-ready report** — exhaustive scholarly entry synthesising every metadata field and image observation |
| 8 | `image_paths` | string (JSON list) | Relative paths to all artifact photographs, e.g. `["images/00001/000016965.jpg", ...]` |
| 9 | `metadata` | string (JSON) | Full structured metadata dict used for generation (see below) |
| 10 | `label` | string | Catalog label from the excavation, e.g. *"PC 20240089"*, *"DT# 2843"* |
| 11 | `item_uuid_hex` | string | Open Context UUID hex for cross-referencing the source API |
| 12 | `num_images` | int | Number of images available for this artifact |
| 13 | `period` | string | Chronological period, e.g. *"Archaic (580 BCE - 482 BCE)"* |
| 14 | `project` | string | Excavation project name, e.g. *"Murlo"*, *"Domuztepe Excavations"* |

### Description Levels Explained

```
Level 1:  "Cover Tile Fragment"
             │
Level 2:  "Terracotta cover tile fragment"
             │
Level 3:  "A wedge-shaped terracotta cover tile fragment with a warm reddish-brown
           surface (Munsell 2.5YR 5/6 Red). The fragment shows smooth exterior..."
             │
Level 4:  "This architectural terracotta cover tile fragment from the Murlo
           excavation (Trench T106, Grid X:211.05 Y:-37.97) measures 14.4 cm
           max preserved length × 10.3 cm width × 2.1 cm thickness..."
             │
Level 5:  "Catalog: PC 20240089. Architectural cover tile fragment, Murlo,
           Poggio Civitate (Etruscan), Archaic period (580–482 BCE). Terracotta,
           Munsell 2.5YR 5/6 Red. Max. pres. length 0.14431 m, width 0.10321 m,
           thickness 0.02134 m. Finger-incised decoration (siglum/sigla) on
           interior surface, 0.023 m in length..."
```

### Metadata Fields (inside `metadata` JSON)

```python
import json
meta = json.loads(row["metadata"])
```

| Key | Meaning | Example |
|---|---|---|
| `label` | Catalog label | `"PC 20240089"` |
| `description` | Original free-text description | `"Body fragment, Impasto..."` |
| `catalog_note` | Short catalog header note | `"2024 season finds"` |
| `object_type` | Controlled vocabulary type | `"Architectural::Cover Tile"` |
| `material` | Material category | `"Terracotta"`, `"Impasto"` |
| `color_munsell` | Verbatim Munsell color value | `"2.5YR 5/6 Red"` |
| `size` | All preserved dimensions | `"Max. Pres. Length: 0.14431 m, ..."` |
| `decorative_tech` | Decorative technique(s) | `"Finger Incised; Siglum/Sigla"` |
| `trench` | Excavation trench designation | `"T106"` |
| `grid_x`, `grid_y` | Grid coordinates within trench | `"211.05"`, `"-37.97"` |
| `elevation` | Elevation in metres | `"598.32"` |
| `date_cataloged` | ISO date artifact was cataloged | `"2024-07-15"` |
| `period` | Chronological period string | `"Archaic (580 BCE - 482 BCE)"` |
| `date_start`, `date_end` | Numeric start/end dates | `-580`, `-482` |
| `coordinates` | Geographic coordinates | `"[11.286, 43.156]"` |
| `project_label` | Excavation project name | `"Murlo"` |
| `latitude`, `longitude` | Site lat/lon | `43.156`, `11.286` |

---

## Use Cases

### 1. CLIP-Style Contrastive Training (Image ↔ Text)

Train a contrastive model to align artifact images with their descriptions. Each artifact provides multiple natural training pairs at different granularities.

#### Basic: Image + Level 5 Pairs

```python
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMAGES_ROOT = Path("/data/group_data/dei-group/archaia")

class ArchaiaCLIPDataset(Dataset):
    """Image-text pairs for CLIP-style contrastive training."""

    def __init__(self, hf_dataset, level="level_5_description", transform=None):
        self.ds = hf_dataset
        self.level = level
        self.transform = transform

        # Expand: one pair per (image, description) for each artifact
        self.pairs = []
        for idx in range(len(self.ds)):
            row = self.ds[idx]
            text = row.get(self.level, "")
            if not text or not text.strip():
                continue
            rel_paths = json.loads(row.get("image_paths", "[]"))
            for rp in rel_paths:
                abs_path = IMAGES_ROOT / rp
                if abs_path.exists():
                    self.pairs.append((str(abs_path), text.strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, text


# Usage
from datasets import load_from_disk

ds = load_from_disk("/data/group_data/dei-group/archaia/archaia_hf_final")
clip_dataset = ArchaiaCLIPDataset(ds, level="level_5_description")
print(f"Total image-text pairs: {len(clip_dataset)}")
# ~90,000+ pairs (31K artifacts × ~3 images each)

image, text = clip_dataset[0]
print(f"Image size: {image.size}")
print(f"Text: {text[:100]}...")
```

#### Advanced: Multi-Level Pairs with Curriculum

Use all 5 levels progressively — train on short descriptions first (easy alignment), then fine-tune on longer ones (harder, more informative):

```python
class ArchaiaMultiLevelDataset(Dataset):
    """
    Multi-level training: each image paired with descriptions at ALL levels.
    Yields (image, text, level) tuples.

    Training strategy:
      - Epoch 1-5:   level_1 + level_2 (easy, short)
      - Epoch 6-10:  level_3 (medium)
      - Epoch 11-20: level_4 + level_5 (hard, detailed)
    """

    LEVELS = [
        "level_1_description",
        "level_2_description",
        "level_3_description",
        "level_4_description",
        "level_5_description",
    ]

    def __init__(self, hf_dataset, levels=None, transform=None):
        self.transform = transform
        levels = levels or self.LEVELS

        self.pairs = []
        for idx in range(len(hf_dataset)):
            row = hf_dataset[idx]
            rel_paths = json.loads(row.get("image_paths", "[]"))
            for rp in rel_paths:
                abs_path = IMAGES_ROOT / rp
                if not abs_path.exists():
                    continue
                for lvl in levels:
                    text = row.get(lvl, "")
                    if text and text.strip():
                        self.pairs.append((str(abs_path), text.strip(), lvl))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, text, level = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, text, level


# Curriculum training
easy   = ArchaiaMultiLevelDataset(ds, levels=["level_1_description", "level_2_description"])
medium = ArchaiaMultiLevelDataset(ds, levels=["level_3_description"])
hard   = ArchaiaMultiLevelDataset(ds, levels=["level_4_description", "level_5_description"])
```

#### With HuggingFace `transformers` CLIP Fine-Tuning

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create a batch
images, texts = [], []
for i in range(8):
    img, txt = clip_dataset[i]
    images.append(img)
    texts.append(txt)

inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image   # (8, 8)

# Contrastive loss: diagonal should have highest similarity
labels = torch.arange(len(images))
loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
loss_t = torch.nn.functional.cross_entropy(logits_per_image.T, labels)
loss = (loss_i + loss_t) / 2

loss.backward()
print(f"Contrastive loss: {loss.item():.4f}")
```

### 2. Image Captioning

Use `(images, level_N_description)` pairs to fine-tune a captioning model:

```python
# Each artifact → caption at desired granularity
for idx in range(len(ds)):
    row = ds[idx]
    image_paths = json.loads(row["image_paths"])
    caption = row["level_3_description"]      # or any level
    # Feed to your captioning model (BLIP-2, LLaVA, etc.)
```

### 3. Archaeological Object Classification

Use `level_1_description` as class labels:

```python
from collections import Counter

labels = [row["level_1_description"] for row in ds]
label_counts = Counter(labels)
print(f"Unique classes: {len(label_counts)}")
for label, count in label_counts.most_common(10):
    print(f"  {count:6d} × {label}")
```

### 4. Metadata-Conditioned Generation

Use the structured metadata to condition text generation:

```python
import json

row = ds[100]
meta = json.loads(row["metadata"])
prompt = f"""
Object type: {meta.get('object_type', 'unknown')}
Material: {meta.get('material', 'unknown')}
Munsell color: {meta.get('color_munsell', 'unknown')}
Dimensions: {meta.get('size', 'unknown')}
Period: {meta.get('period', 'unknown')}
Site: {meta.get('project_label', 'unknown')}

Write a detailed archaeological description of this artifact.
"""
```

---

## Data Sources

Artifacts are drawn from [Open Context](https://opencontext.org/), a peer-reviewed open-access repository for archaeological data. Projects included:

| Project | Region | Period |
|---|---|---|
| Murlo (Poggio Civitate) | Tuscany, Italy | Etruscan (Archaic) |
| Domuztepe Excavations | Turkey | Neolithic / Chalcolithic |
| The Gabii Project | Latium, Italy | Roman |
| Pyla-Koutsopetria Archaeological Project | Cyprus | Classical – Medieval |
| The Eastern Korinthia Archaeological Survey | Peloponnese, Greece | Multi-period |

---

## Generation Details

| Parameter | Value |
|---|---|
| Model | GPT-4o-mini |
| Image detail | `high` (full-resolution tokens) |
| Max tokens | 2048 |
| Temperature | 0.7 |
| Images per artifact (for generation) | up to 3 |
| Total artifacts | 31,624 (31,605 after cleaning) |

Descriptions were generated by sending artifact photographs plus all structured metadata to the model with a detailed prompt specifying content requirements for each hierarchical level. No word-count constraints were imposed — length is determined by the amount of evidence available.