# Satellite Land-Use Classifier & Temporal Change Detector

A computer vision system that classifies satellite tiles into 10 land-use
categories and detects temporal change between two time periods, using
transfer learning (ResNet-18), embedding-based cosine similarity, and a
Streamlit dashboard.

---

## Demo Video

**[Watch the 3-minute demo](https://drive.google.com/file/d/1eTjhYf5pP79c9rLJEkJqowUoZQVlNs9y/view?usp=sharing)**

---

## Overview

| Module | What it does |
|---|---|
| **1 — Land-Use Classifier** | Two-phase fine-tuned ResNet-18 (frozen head → unfrozen last 2 blocks), evaluated on an EuroSAT holdout and on UC Merced as a cross-domain generalization test |
| **2 — Temporal Change Detector** | Reuses the fine-tuned backbone as a feature extractor; cosine similarity between embeddings + ROC-derived thresholds flags land-use change between tile pairs |
| **3 — Geo-Dashboard** | Local, offline Streamlit app: upload before/after tiles, get per-tile classification, similarity score, change flag, and a visual diff heatmap |

---

## Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| **EuroSAT RGB** | Primary — training/val/test | 27,000 tiles, 10 classes, 64×64 | [Zenodo](https://zenodo.org/records/7711810) / Kaggle mirror `nilesh789/eurosat-rgb` |
| **UC Merced Land Use** | Holdout — cross-domain evaluation only, never trained on | 2,100 images, 21 classes, 256×256 | Kaggle mirror `abdulhasibuddin/uc-merced-land-use-dataset` |

Raw datasets are **not committed to this repo** (too large, and redundant
with Kaggle). To reproduce:
- On Kaggle: attach both via **+ Add Data** in the notebook sidebar.
- Locally (only needed for `data/sample_pairs/` demo images): download a
  handful of tiles directly from either Kaggle dataset page.

---

## Repository Structure

```
satellite-landuse-project/
├── app/
│   └── streamlit_app.py          # Local dashboard (Module 3)
├── notebooks/                     # Run on Kaggle, in order
│   ├── 01_data_pipeline.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_change_detection.ipynb
│   ├── 05_error_analysis.ipynb
│   ├── 06_embedding_viz.ipynb     # Bonus C
│   └── 07_imbalance_exp.ipynb     # Bonus D
├── src/
│   ├── models.py
│   └── ...
├── data/
│   ├── splits/                    # train/val/test CSVs (filenames + labels + cluster_id — no images)
│   └── sample_pairs/              # Small set of demo tiles for the app/GradCAM
├── checkpoints/
│   └── resnet18_finetuned.pt      # Git LFS — required to run the app
├── outputs/
│   ├── figures/ metrics/ confusion_matrices/
│   ├── roc/ heatmaps/ gradcam/ embeddings/
├── run_gradcam_bonus.py           # Bonus A — run locally, no training needed
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and enter the repo
```bash
git clone <your-repo-url>
cd satellite-landuse-project
git lfs install
git lfs pull          # fetches resnet18_finetuned.pt
```

### 2. Python environment
Python **3.10**.
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Reproduce training (optional — checkpoints are already included)
On Kaggle: create a notebook, enable GPU, attach both datasets (see
Datasets section above), and run `notebooks/01` through `07` in order.

### 4. Run the dashboard
Requires `checkpoints/resnet18_finetuned.pt` and
`outputs/metrics/change_detection_thresholds.csv` (both included via Git
LFS / repo).
```bash
streamlit run app/streamlit_app.py
```
Opens at `http://localhost:8501`. Runs fully offline once dependencies
are installed — the backbone loads with `weights=None` so it never
fetches ImageNet weights at runtime.

### 5. Run GradCAM (Bonus A)
```bash
python run_gradcam_bonus.py
```
Automatically uses every image in `data/sample_pairs/`. Outputs saved to
`outputs/gradcam/`.

---

## Key Methodological Decisions

- **Spatial split without real coordinates.** EuroSAT_RGB carries no
  per-tile geolocation (only the raw Sentinel-2/GeoTIFF version does).
  The train/val/test split instead uses **per-class KMeans clustering on
  ImageNet-pretrained ResNet-18 embeddings** as a documented
  visual-similarity proxy for spatial locality, with cluster→split
  assignment proportioned by tile count (not raw cluster count, since
  clusters are unequal in size).
- **Two-phase fine-tuning**, per spec: frozen backbone + head only
  (3 epochs, lr 1e-3) → unfreeze `layer3`/`layer4` with the optimizer
  rebuilt (5 epochs, lr 1e-4). Best checkpoint selected by validation
  loss.
- **UC Merced mapping is documented, not exhaustive** — classes with a
  poor semantic fit to any EuroSAT category are excluded from evaluation
  rather than force-mapped (see `src/` mapping definition and the report
  for the excluded list and reasoning).
- **Change-detection pairs are deliberately non-trivial.** Unchanged
  pairs are sampled from the *same class and same KMeans cluster*
  (visually/spatially consistent), not just the same class. Changed
  pairs are sampled from a **plausible-transition adjacency map**
  (e.g. Forest↔HerbaceousVegetation) rather than arbitrary class pairs —
  random cross-class pairing would be trivially easy and inflate ROC-AUC
  without testing anything meaningful.
- **Embeddings for change detection come from the fine-tuned backbone**,
  not the ImageNet one used for the spatial split — the two serve
  different purposes and are kept separate to avoid circularity.
- **The heatmap is a visualization aid only.** The change decision is
  made exclusively from cosine similarity between embeddings compared
  against a chosen operating threshold — never from the pixel-difference
  heatmap.

---

## Results

### Classification

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline CNN (scratch, 64×64) | 71.12% | 70.67% | 75.43% | 73.68% |
| ResNet-18 — Phase 1 (frozen) | 70.38% | 69.83% | — | — |
| ResNet-18 — Phase 2 (fine-tuned) | **88.80%** | **88.61%** | **92.39%** | **92.26%** |

| Cross-domain evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| UC Merced holdout (fine-tuned ResNet-18) | 39.6% | 33.52% |

The sharp drop on UC Merced (92.3% EuroSAT test → 33.5% UC Merced) is a
genuine cross-domain generalization finding, not a bug — see
**Limitations** below.

Full per-class F1 and confusion matrices: `outputs/metrics/*_per_class_f1.csv`,
`outputs/confusion_matrices/`.

### Temporal Change Detection

- **ROC-AUC: 0.9791** on 2,000 evaluated pairs (1,000 unchanged + 1,000
  changed, sampled from held-out val+test tiles only).

  | Mode | Threshold (change score) | TPR | FPR |
  |---|---:|---:|---:|
  | Balanced (Youden's J) | 0.333 | 0.909 | 0.059 |
  | High recall (TPR≥0.90) | 0.338 | 0.900 | 0.052 |
  | High precision (FPR≤0.05) | 0.342 | 0.897 | 0.050 |

  The three thresholds cluster tightly (span of 0.009) because the
  embedding space separates changed/unchanged pairs very cleanly — a
  consequence of the high AUC, not a flaw in threshold selection. Only
  ~1.1% of evaluated pairs (22/2,000) fall in the band where operating-point
  choice changes the outcome.

### Spatial Leakage Experiment (random split vs. block split)

See `outputs/figures/05_spatial_leakage_experiment.png` for the
random-vs-block accuracy/macro-F1 comparison. *(Exact values pending —
the source CSV wasn't included in the outputs archive; add the numbers
here once available.)*

### Bonus C — Embedding Visualization (t-SNE/UMAP)

All 27,000 EuroSAT embeddings projected to 2D via UMAP, comparing
**ImageNet-pretrained ResNet-18** embeddings against the **fine-tuned**
ResNet-18 embeddings, colored by class. See
`outputs/figures/06_embedding_umap_comparison.png` and coordinates in
`outputs/metrics/06_umap_coordinates.csv`.

### Bonus D — Imbalance Experiment

`Pasture` and `PermanentCrop` downsampled to 20% of their original
training size, retrained, then compared against a weighted-loss
mitigation:

| Setting | Val Accuracy | Val Macro-F1 |
|---|---:|---:|
| Balanced (original) | 88.80% | 88.61% |
| Imbalanced, no mitigation | 89.52% | 89.32% |
| Imbalanced + weighted loss | 88.32% | 88.10% |

| Downsampled class | Balanced F1 | Imbalanced (no mitigation) F1 | Imbalanced (weighted loss) F1 |
|---|---:|---:|---:|
| Pasture | 96.40% | 97.32% | 92.33% |
| PermanentCrop | 78.57% | 75.36% | 72.71% |

**Note — counter to the typical expectation:** weighted loss did **not**
recover F1 on the downsampled classes here; both `Pasture` and
`PermanentCrop` score *lower* under weighted loss than under no
mitigation at all, and overall macro-F1 also dropped slightly relative
to the unmitigated imbalanced run. Reported as observed rather than
smoothed to the expected textbook outcome — see the report for
discussion (plausible causes: 20% downsampling still leaves a sizeable
sample count for these two classes, and inverse-frequency weighting can
overcorrect and destabilize training on an already well-separated
dataset).

### Bonus A — GradCAM

10 interpreted examples (2 each across AnnualCrop, Forest, Industrial,
Residential, SeaLake) in `outputs/gradcam/`, exceeding the brief's
minimum of 3.

### Error Analysis

Top-5 highest-confidence UC Merced misclassifications
(`outputs/metrics/05_top5_classifier_errors_uc_merced.csv`):

| Original UC Merced class | Mapped true class | Predicted | Confidence |
|---|---|---|---:|
| mobilehomepark | Residential | Industrial | 100.0% |
| river | River | Forest | 99.999% |
| forest | Forest | Residential | 99.994% |
| chaparral | HerbaceousVegetation | Forest | 99.987% |
| agricultural | AnnualCrop | Residential | 99.947% |

All five are wrong with near-total confidence — see **Limitations** for
what this indicates about model calibration on out-of-domain inputs.
Visual interpretation and hypotheses for each: see report.

Top-5 change-detector errors: `outputs/metrics/05_top5_change_detector_errors.csv`.

---

## Limitations

- **Spatial split is a visual-similarity proxy, not literal geography** —
  EuroSAT_RGB doesn't expose real coordinates; documented and justified
  above rather than worked around by switching datasets.
- **UC Merced domain gap is real and large** (88.6% EuroSAT val → 33.5%
  UC Merced macro-F1). A resolution/blur-matching test was run
  specifically to isolate the cause; it did **not** meaningfully improve
  results, indicating the gap is more likely sensor/imaging-domain
  driven (orbital Sentinel-2 vs. low-altitude aerial photography) than
  purely a resolution artifact.
- **The model is overconfident on out-of-domain errors, not just wrong.**
  All 5 of its highest-confidence UC Merced misclassifications (see
  Error Analysis above) are wrong with ≥99.9% confidence — the model
  shows no calibration uncertainty even on visually ambiguous or
  domain-shifted tiles, a distinct failure mode from simply "getting it
  wrong."
- **UC Merced mapping covers a subset of its 21 classes** — weak
  semantic fits (e.g. sports/recreation facilities) were excluded from
  evaluation rather than force-mapped.
- **Temporal change is simulated**, not from real before/after imagery —
  T1/T2 pairs are constructed from the same static EuroSAT collection
  using class-transition semantics, explicitly documented as such.
- **Change-detection AUC partly reflects classification separability.**
  Because "changed" pairs are, by construction, cross-class, and the
  backbone was trained with cross-entropy to separate classes, the
  0.98 AUC is partly a restatement of classification quality — the
  detector's performance on subtler, same-class change is untested by
  this evaluation.

---

## Requirements

Python 3.10. Full dependency list in `requirements.txt` — covers both
the Kaggle notebook stack (torch, scikit-learn, matplotlib, etc.) and the
local dashboard (streamlit).

---

## Data Attribution

- EuroSAT: Helber et al., *"EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification"*.
- UC Merced Land Use: Yang & Newsam, *"Bag-of-visual-words and spatial
  extensions for land-use classification"*.

Both accessed via the Kaggle mirrors linked in the Datasets section.