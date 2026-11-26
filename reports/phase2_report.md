# Phase 2 Report — Data Preparation & Baseline Model Development

**Course:** SBES426 — Medical Pattern Recognition (Cairo University, Fall 2026)  
**Team:** Team 2 — Farah Ahmed, Malak Mohamed, Reem Khaled, Omar Nasser, and 5th member  
**Supervisor:** Eng. Mary Milad  
**Project:** Obstetric Ultrasound Segmentation and Fetal Head Circumference (HC) Measurement

---

## 1. Data Cleaning & Preprocessing Pipeline
- **Duplicate Resolution:** Document the 3 exact duplicate file groups identified in Phase 1 EDA and how they were cleaned prior to splitting.
- **Mask Generation:** Describe flood-filling the hollow `*_Annotation.png` ellipse contours into solid binary segmentation masks (`1` = fetal head, `0` = background).
- **Dimension Standardization & Pixel Calibration:** Document target resolution, handling of the 48 non-800×540 images, and effective `mm/px` recalibration.
- **Intensity Normalization & Edge-Preserving Speckle Filtering:** Specify normalization range and despeckling filter parameters.

---

## 2. Leakage-Free Data Splitting Strategy
- **Grouping:** Exam-level numeric prefix grouping (`000`–`805`) so multi-plane scans (e.g., `010_HC.png` and `010_2HC.png`) never span across splits.
- **Stratification:** Stratified across HC size/trimester bins to maintain balanced distributions across Train (70%), Validation (15%), and Internal Test (15%).
- **Split Summary Table:** *(Populate after running `02_preprocessing_and_splitting.ipynb`)*

| Split | Images | Share (%) | Unique Exams | Mean HC (mm) | SD HC (mm) |
|---|---:|---:|---:|---:|---:|
| Train | — | — | — | — | — |
| Validation | — | — | — | — | — |
| Internal Test | — | — | — | — | — |
| **Total** | — | 100.0% | — | — | — |

---

## 3. Imbalance Handling
- **Pixel-Level Class Imbalance:** Report foreground vs. background pixel ratio and loss formulation (**Dice Loss + Weighted BCE**).
- **Trimester / Size Imbalance:** Stratified splitting + training-only augmentations (flips, rotations, brightness/contrast jitter).

---

## 4. Baseline Model Architecture & Training Settings
- **Architecture:** *(e.g., Lightweight Baseline U-Net — encoder/decoder channels, activation, normalization, parameter count)*
- **Training Configuration:** Loss function, optimizer, learning rate, scheduler, batch size, epochs, random seed (`42`), hardware.
- **Postprocessing:** Connected-component filtering, contour extraction, least-squares ellipse fitting (`cv2.fitEllipse`), and physical HC (mm) calculation via Ramanujan's formula.

---

## 5. Evaluation Results & Comparison with Reference Baselines

### 5.1 Pixel-Level Classification Metrics (Internal Test Set)
- Confusion Matrix (`reports/figures/confusion_matrix.png`)
- Sensitivity (Recall): `—`
- Specificity: `—`
- F1-Score (DSC): `—`
- AUC-ROC: `—`

### 5.2 Biometric Metrics & Literature Comparison
| Method | DSC (%) | HD (mm) | AD (mm) | MD (mm) | Notes |
|---|---:|---:|---:|---:|---|
| van den Heuvel et al. (2018) | 97.00% | — | 2.80 | — | Random Forest + Hough + Ellipse fitting (HC18 test) |
| Sobhaninia et al. (2019) | 96.84% | — | 2.12 | — | Multi-task Link-Net CNN (HC18 test) |
| **Our Baseline (Raw/Normalized)** | — | — | — | — | Internal Test Split |
| **Our Baseline (Despeckled)** | — | — | — | — | Internal Test Split |

---

## 6. Limitations & Planned Improvements for Phase 3
- **Current Limitations:** *(Acoustic shadowing failures, 1st/3rd trimester boundary errors, resize aspect/calibration sensitivity, baseline model capacity.)*
- **Phase 3 Plan:**
  1. Controlled ablation across edge-preserving despeckling filters (Bilateral, SRAD, Anisotropic Diffusion, Non-Local Means).
  2. Upgraded segmentation architectures (e.g., Attention U-Net / ResNet-encoder U-Net / Link-Net) and ellipse-aware postprocessing.
  3. Clinician GUI comparing predicted HC (mm) and gestational age against **Verburg et al. (2008)** 10th/90th percentile reference growth charts.
