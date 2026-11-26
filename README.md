# Obstetric Ultrasound Segmentation and Fetal Head Circumference (HC) Measurement

**Course:** SBES426 — Medical Pattern Recognition (Cairo University, Faculty of Engineering, Fall 2026)  
**Team:** Team 2 — Farah Ahmed, Malak Mohamed, Reem Khaled, Omar Nasser, and our 5th team member  
**Supervisor:** Eng. Mary Milad  

---

## Project Overview

Accurate measurement of fetal **Head Circumference (HC)** from 2D standard-plane obstetric ultrasound images is essential for estimating gestational age, monitoring fetal growth, and screening for abnormalities such as intrauterine growth restriction (IUGR), microcephaly, and macrocephaly. Manual caliper tracing suffers from inter- and intra-observer variability caused by acoustic speckle noise, low soft-tissue contrast, and cranial shadowing.

### Research Question
> Does **edge-preserving speckle-reduction preprocessing** experimentally improve automated fetal head segmentation overlap and physical HC measurement accuracy (in mm) compared with raw/normalized ultrasound images?

### Reference Baselines
| Reference | Method | Dice (DSC %) | Mean Absolute HC Error (mm) |
|---|---|---:|---:|
| van den Heuvel et al. (2018) | Haar-like features + Random Forest + Hough transform + Ellipse fitting | 97.00% | 2.80 mm |
| Sobhaninia et al. (2019) | Multi-task Link-Net CNN + Ellipse parameter estimation | 96.84% | 2.12 mm |

---

## Dataset

**HC18 Grand Challenge Dataset** — [Zenodo Record 1322001](https://zenodo.org/records/1322001)
- **999 labeled training images** (`*_HC.png`, predominantly 800×540 px grayscale; 48 images have varying dimensions) + **999 expert hollow ellipse contour annotations** (`*_Annotation.png`).
- **335 unannotated official test images** (`test_set/`).
- Metadata CSVs:
  - `training_set_pixel_size_and_HC.csv` (`filename`, `pixel size`, `head circumference (mm)`)
  - `test_set_pixel_size.csv` (`filename`, `pixel size`)
- **Phase 1 EDA highlights:**
  - 0 missing values across all 999 CSV rows; 100% filename match.
  - 3 exact duplicate file groups identified for deduplication prior to splitting.
  - HC range: **44.30 – 346.40 mm** (mean 174.38 mm, median 174.06 mm, SD 65.28 mm); IQR-flagged extremes correspond to valid 1st and 3rd trimester scans and are retained.
  - Pixel size range: **0.0494 – 0.3933 mm/px** (mean 0.1398 mm/px, SD 0.0530 mm/px).

> Raw dataset images and archives under `data/raw/` are gitignored due to size. Download the dataset from Zenodo and place extracted folders and CSVs in `data/raw/`.

---

## Repository Structure

```text
Obstetric-Ultrasound-Segmentation-and-Measurement/
├── data/
│   ├── raw/                  # Raw HC18 images, CSVs, and archives/ (gitignored)
│   ├── interim/              # Deduplicated metadata and flood-filled solid binary masks (gitignored)
│   ├── processed/            # Resized, normalized, and despeckled images/masks (gitignored)
│   └── splits/               # Leakage-free exam-grouped stratified train/val/test CSV manifests
├── notebooks/
│   ├── 01_phase1_eda.ipynb                   # Archival Phase 1 Exploratory Data Analysis
│   ├── 02_preprocessing_and_splitting.ipynb  # Deduplication, mask fill, despeckling, grouped split
│   ├── 03_baseline_training.ipynb            # Baseline segmentation model training & loss tuning
│   └── 04_evaluation_and_comparison.ipynb    # Pixel & biometric evaluation vs. literature baselines
├── src/
│   ├── config.py             # Centralized paths, column names, image dimensions, random seed
│   ├── data_prep/            # Duplicate cleaning, contour flood-fill, exam-grouped stratified split
│   ├── preprocessing/        # Dimension standardization (+ mm/px scaling), normalization, despeckling
│   ├── models/               # Baseline segmentation architectures and Dice + Weighted BCE losses
│   ├── postprocessing/       # Contour extraction, cv2.fitEllipse, Ramanujan HC (mm)
│   ├── evaluation/           # Confusion matrix, Sensitivity, Specificity, F1/DSC, AUC-ROC, HD, AD, MD
│   └── clinical/             # Verburg et al. (2008) 10th/90th percentile reference charts & clinician GUI
├── reports/
│   ├── figures/              # Generated plots, overlays, and confusion matrices
│   ├── phase1_report.pdf     # Phase 1 written report
│   ├── phase1_presentation.pdf / .pptx
│   ├── hc18_csv_schema_summary.csv
│   ├── hc18_image_dimensions.csv
│   └── phase2_report.md      # Phase 2 report template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Project Roadmap

1. **Phase 1 (Completed):** Problem formulation, literature review, and HC18 EDA (`notebooks/01_phase1_eda.ipynb`, `reports/phase1_report.pdf`).
2. **Phase 2 (In Progress):**
   - Clean 3 exact duplicate groups, flood-fill hollow ellipse annotations into solid binary masks (`1`=head, `0`=background), standardize dimensions with calibrated `mm/px` adjustment, normalize intensities, and apply edge-preserving speckle filtering.
   - Leakage-free **70% / 15% / 15%** Train / Validation / Internal Test split grouped by exam numeric prefix ID and stratified across HC size/trimester bins.
   - Handle foreground/background pixel imbalance (Dice Loss + Weighted BCE) and trimester distribution imbalance (stratified split + geometric/intensity augmentations).
   - Train baseline segmentation model, fit least-squares ellipses (`cv2.fitEllipse`), and compute physical HC via Ramanujan's perimeter approximation.
   - Report pixel metrics (Confusion Matrix, Sensitivity, Specificity, F1/DSC, AUC-ROC) and biometric metrics (DSC %, HD mm, AD mm, MD mm) vs. van den Heuvel et al. (2018) and Sobhaninia et al. (2019).
3. **Phase 3 (Planned):** Systematic despeckling ablation study, advanced architectures, and a clinician GUI comparing predicted HC (mm) + gestational age against Verburg et al. (2008) 10th/90th percentile reference growth charts.
