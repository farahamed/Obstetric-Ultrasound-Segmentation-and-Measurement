# Phase 2 Report — Data Preparation, Preprocessing, and Baseline Model Development

**Course:** SBES426 — Medical Pattern Recognition (Cairo University, Faculty of Engineering, Fall 2026)  
**Team:** Team 2 — Farah Ahmed, Malak Mohamed, Reem Khaled, Omar Nasser, and Yousra Hatem  
**Supervisor:** Eng. Mary Milad  
**Project:** Obstetric Ultrasound Segmentation and Fetal Head Circumference (HC) Measurement  

---

## 1. Data Cleaning & Preprocessing Pipeline

Building on the Phase 1 Exploratory Data Analysis of the **HC18 Grand Challenge** dataset (999 labeled ultrasound scans, 999 expert hollow ellipse annotations, and 335 unannotated test images), we implemented a reproducible modular data preparation and preprocessing pipeline in `src/data_prep/` and `src/preprocessing/` (executed via `notebooks/02_preprocessing_and_splitting.ipynb`).

### 1.1 Duplicate Detection & Resolution
All 999 ultrasound images (`*_HC.png`) and 999 annotations (`*_Annotation.png`) were audited using **two-level SHA-256 hashing** (raw file bytes and decoded grayscale pixel arrays) with Union-Find clustering. Beyond the **3 exact file-hash duplicate groups** flagged in Phase 1, decoded-pixel hashing uncovered a **4th pixel-identical annotation duplicate** (`306_2HC.png` and `308_HC.png`, whose PNG headers differed while the annotation arrays were 100% identical and the ultrasound mean absolute difference was `0.0439 / 255`). Within each group, one canonical primary scan was retained and redundant copies were removed prior to splitting (`999 -> 994` unique cleaned images):

| Group | Match Type | Kept Scan | Dropped Redundant Scan(s) | Pixel Size (mm/px) | HC (mm) | Ultrasound MAD vs. Kept |
|---:|---|---|---|---:|---:|---:|
| 1 | Exact pixel annotation | `308_HC.png` | `306_2HC.png` | 0.175531 | 165.68 | 0.0439 |
| 2 | Exact file annotation | `484_HC.png` | `482_2HC.png`, `483_3HC.png` | 0.145447 | 174.99 | 0.7133 (`482`/`483` exact 0.0) |
| 3 | Exact file annotation | `630_HC.png` | `629_2HC.png` | 0.130545 | 166.73 | 0.1692 |
| 4 | Exact file annotation | `978_3HC.png` | `979_4HC.png` | 0.261567 | 312.49 | 0.0935 |

All valid 1st- and 3rd-trimester HC extremes flagged by IQR in Phase 1 (range `44.30 – 346.40 mm`) were intentionally preserved because they represent genuine gestational stages rather than measurement errors.

### 1.2 Hollow Ellipse Contour to Solid Binary Mask Conversion
Expert HC18 annotations are stored as 1-pixel hollow white ellipses on black backgrounds. We converted all 994 cleaned annotations into solid binary segmentation masks (`0` = background, `1` = fetal head):
- **990 masks** were filled directly via `scipy.ndimage.binary_fill_holes`.
- **4 border-clipped annotations** (`231_HC`, `427_2HC`, `776_HC`, `977_HC`) intersected the top or bottom image boundary, forming open C-arcs that remain hollow under standard flood-fill. Our pipeline automatically detected these (`filled_px <= 2 * contour_px`) and repaired them by fitting the exact ellipse (`cv2.fitEllipse`) to the white contour coordinates and rasterizing the filled interior clipped cleanly at the image boundary (`reports/figures/data_prep_mask_verification.png`).

### 1.3 Spatial Standardization & Anisotropic Pixel-Size Recalibration
- Because 24 labeled pairs deviate from the majority `800 × 540` resolution, all 994 images (`cv2.INTER_AREA`) and solid masks (`cv2.INTER_NEAREST`) were standardized to **`256 × 384`** (aspect ratio `1.5`, matching native `800 / 540 = 1.4815`).
- Exact per-axis physical spacings were recalibrated as:
  \[
  p_x' = p_{\text{orig}} \cdot \frac{W_{\text{orig}}}{W_{\text{target}}}, \qquad
  p_y' = p_{\text{orig}} \cdot \frac{H_{\text{orig}}}{H_{\text{target}}}
  \]
- **Calibration Validation (all 994 resized masks):** Fitting ellipses to the resized `256 × 384` masks in recalibrated millimeter space recovered CSV ground-truth HC with **MAE = `0.285 mm`**, **Median |Err| = `0.261 mm`**, and **95th percentile = `0.515 mm`** (`reports/figures/preprocessing_calibration_check.png`).

### 1.4 Intensity Normalization & Edge-Preserving Speckle Reduction
- **Normalization:** Applied `robust_minmax` normalization (`[0.5, 99.5]` percentile scaling computed over non-zero active ultrasound sector pixels while preserving pure-black padding borders at `0`).
- **Edge-Preserving Despeckling Benchmark:** To investigate our research question experimentally without touching the test split, we implemented and benchmarked **Bilateral**, **Perona-Malik Anisotropic Diffusion** (`n_iter=12, kappa=18.0, gamma=0.15`), and **Non-Local Means (NLM)** exclusively on the **Validation split (`n = 148`)**:

| Filter | ENL (↑ smoother) | CNR (↑ contrast) | EPI (↑ edge fidelity) | Interior Speckle Var Reduction | Runtime (ms/img) |
|---|---:|---:|---:|---:|---:|
| `raw_normalized` *(Control Arm)* | 1.897 | 0.5679 | 1.0000 | 0.00% | 0.00 |
| `bilateral` | 2.054 | 0.5667 | 0.8302 | 7.08% | 5.40 |
| `anisotropic_diffusion` *(Experimental Arm)* | **2.085** | 0.5645 | 0.6699 | **7.49%** | 20.65 |
| `nlm` | 1.982 | **0.5722** | **0.9168** | 5.01% | 176.26 |

Two parallel processed datasets (`images_raw_norm/` and `images_despeckled/` using Perona-Malik Anisotropic Diffusion) were exported so our baseline model could be trained identically on both arms (`reports/figures/preprocessing_filter_comparison.png`).

---

## 2. Leakage-Free Data Splitting Strategy

- **Multi-Plane Exam Grouping:** Auditing the filenames revealed that consecutive multi-plane scans from the same exam appear as runs of `*_2HC.png`, `*_3HC.png`, `*_4HC.png`, `*_7HC.png` immediately preceding their primary `*_HC.png` scan. Reconstructing **`779` unique exam groups** (`exam_group_id`, within-group median HC spread = `4.55 mm`) ensured that sibling ultrasound planes from the same examination never leaked across Train, Validation, and Test.
- **HC-Stratified Splitting (`70% / 15% / 15%`, `RANDOM_SEED = 42`):** Exams were stratified across 6 clinical HC size strata (`<100`, `100–150`, `150–200`, `200–250`, `250–300`, `>300` mm). Automated assertions verified **zero overlap** across all splits by `exam_group_id`, `filename`, `us_pixel_sha256`, and `ann_pixel_sha256` (`reports/figures/data_prep_split_distribution.png`).

| Split | Images | Share (%) | Unique Exams | Mean HC (mm) | Median HC (mm) | SD HC (mm) | Mean Pixel Size (mm/px) | FG Pixels (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 693 | 69.72% | 545 | 173.93 | 174.10 | 64.89 | 0.13911 | 30.59% |
| Validation | 148 | 14.89% | 117 | 173.40 | 173.09 | 61.46 | 0.13813 | 30.52% |
| Internal Test | 153 | 15.39% | 117 | 176.59 | 173.96 | 70.90 | 0.14379 | 29.44% |
| **Total** | **994** | **100.00%** | **779** | **174.26** | **174.06** | **65.34** | **0.13968** | **30.41%** |

---

## 3. Imbalance Handling

1. **Pixel-Level Class Imbalance (`30.59%` Foreground vs. `69.41%` Background on Train):**
   - Trained with a combined **Soft Dice Loss + Class-Weighted BCEWithLogitsLoss** (`0.5 * SoftDice + 0.5 * WeightedBCE`), using positive class weight `pos_weight = 2.269` computed strictly from the training masks.
2. **Trimester / Size Distribution Imbalance (`54%` of scans lie in `150–200 mm`):**
   - Combined exam-level HC-bin stratification with tempered inverse-frequency stratum sampling (`WeightedRandomSampler`, `power = 0.5`) and synchronized **training-only augmentations**: horizontal flip (`p = 0.5`), vertical flip (`p = 0.2`), affine rotation (`±15°`), mild scale (`±8%`), translation (`±5%`), brightness/contrast jitter (`±15%`), and subtle Gaussian sensor noise (`σ = 0.02`).

---

## 4. Baseline Model Architecture, Training Settings & Postprocessing

### 4.1 Architecture (`src/models/unet.py`)
- **Model:** Lightweight 4-stage 2D **Baseline U-Net** (`491,729` trainable parameters).
- **Channel Progression:** Encoder `1 -> 8 -> 16 -> 32 -> 64`, Bottleneck `128` (with `Dropout2d(0.15)`), Decoder `64 -> 32 -> 16 -> 8 -> 1` using double `3×3 Conv -> BatchNorm2d -> ReLU` blocks, `2×2 MaxPool2d`, bilinear upsampling, and skip connections.

### 4.2 Training Settings (`src/models/trainer.py`, `notebooks/03_baseline_training.ipynb`)
- **Input Resolution:** `128 × 192` grayscale `[0, 1]` (preserving HC18 aspect ratio).
- **Optimizer & Schedule:** `AdamW` (`lr = 2e-3`, `weight_decay = 1e-4`), `CosineAnnealingLR` (`eta_min = 1e-5`), gradient norm clipping (`max_norm = 2.0`).
- **Batch Size / Epochs / Seed:** `batch_size = 16`, `epochs = 18`, `early_stopping_patience = 7`, `RANDOM_SEED = 42`, CPU training (~6.2 min per arm).
- **Validation Convergence:** Both arms peaked cleanly at Epoch 14 without overfitting (`reports/figures/baseline_training_curves.png`):
  - **Raw Normalized (`raw_norm`):** Best Val Dice = **`97.18%`**, Val IoU = **`94.58%`**, Val Loss = `0.05600`.
  - **Despeckled (`despeckled`):** Best Val Dice = **`96.83%`**, Val IoU = **`93.99%`**, Val Loss = `0.05858`.

### 4.3 Postprocessing & Physical HC Measurement (`src/postprocessing/ellipse_fitting.py`)
1. Bilinearly upsample predicted soft probability maps from `128 × 192` back to each scan's **native original resolution** (`~800 × 540`) for exact evaluation against full-resolution expert masks using native `pixel size (mm/px)`.
2. Threshold at `0.5`, apply morphological closing (`7×7` elliptical kernel), fill internal holes (`binary_fill_holes`), and keep the largest connected component.
3. Extract the external boundary contour (`cv2.findContours`), fit a least-squares ellipse (`cv2.fitEllipse`) in physical millimeter coordinates to obtain semi-axes \((a, b)\), and compute physical Head Circumference via **Ramanujan's second approximation**:
   \[
   h = \frac{(a - b)^2}{(a + b)^2}, \qquad
   \mathrm{HC} = \pi (a + b)\left(1 + \frac{3h}{10 + \sqrt{4 - 3h}}\right)
   \]

---

## 5. Evaluation Results & Comparison with Reference Baselines

All metrics below were computed on the strictly held-out **Internal Test Split (`n = 153` unseen images across `117` unseen exams; `66,083,122` native pixels)** via `notebooks/04_evaluation_and_comparison.ipynb`.

### 5.1 Pixel-Level Classification Metrics (Internal Test Set, `n = 153`)

| Input Arm | TN (px) | FP (px) | FN (px) | TP (px) | Sensitivity (%) | Specificity (%) | Precision (%) | F1 / Pixel DSC (%) | Accuracy (%) | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Raw Normalized (`raw_norm`)** | 46,085,316 | 539,399 | 531,157 | 18,927,250 | 97.27% | **98.84%** | **97.23%** | **97.25%** | **98.38%** | **0.99893** |
| **Despeckled (`despeckled`)** | 46,000,799 | 623,916 | 468,657 | 18,989,750 | **97.59%** | 98.66% | 96.82% | 97.20% | 98.35% | 0.99887 |

- Confusion Matrices: `reports/figures/confusion_matrix.png`
- ROC Curves (full & zoomed high-TPR operating region): `reports/figures/roc_curves.png`

### 5.2 Biometric Metrics & Literature Comparison

| Method | Pipeline | Eval Split | DSC (%) | HD (mm) | HD95 (mm) | Mean AD (mm) | Median AD (mm) | Mean MD (mm) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| van den Heuvel et al. (2018) | Haar + Random Forest + Hough + Ellipse fit | Official HC18 Test (`n=335`) | 97.00% | — | — | 2.80 | — | — |
| Sobhaninia et al. (2019) | Multi-task Link-Net CNN + Ellipse Tuner | Official HC18 Test (`n=335`) | 96.84% | — | — | **2.12** | — | — |
| **Our Baseline U-Net (`raw_norm`)** | 4-stage U-Net (492k) + `cv2.fitEllipse` | Internal Test (`n=153`) | **97.12% ± 2.34%** | **1.65 ± 1.01** | **1.53** | **2.45 ± 2.55** | **1.88** | **+1.00 ± 3.40** |
| **Our Baseline U-Net (`despeckled`)** | Perona-Malik + 4-stage U-Net + `cv2.fitEllipse` | Internal Test (`n=153`) | 96.61% ± 8.12% | 1.79 ± 2.03 | 1.67 | 3.27 ± 4.36 | 2.25 | +1.74 ± 5.17 |

*Note:* Direct numeric comparisons with van den Heuvel et al. (2018) and Sobhaninia et al. (2019) should be interpreted cautiously because the official HC18 test set (`n = 335`) does not publicly release ground-truth labels; our evaluation uses our strictly exam-disjoint internal test split (`n = 153`).

#### Trimester-Stratified Biometric Performance (`n = 153`)

| Model | Trimester Stage | Images | DSC (%) | HD (mm) | Mean AD (mm) | Mean MD (mm) | van den Heuvel (2018) AD (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| `BaselineUNet (raw_norm)` | 1st Trimester (`<100 mm`) | 30 | **94.15%** | **1.66** | **2.86** | +0.75 | 3.10 |
| `BaselineUNet (raw_norm)` | 2nd Trimester (`100–250 mm`) | 96 | **97.94%** | **1.34** | **1.89** | +0.87 | 2.40 |
| `BaselineUNet (raw_norm)` | 3rd Trimester (`>250 mm`) | 27 | **97.49%** | **2.74** | **4.00** | +1.75 | 4.80 |
| `BaselineUNet (despeckled)` | 1st Trimester (`<100 mm`) | 30 | 91.89% | 2.11 | 4.05 | -0.48 | 3.10 |
| `BaselineUNet (despeckled)` | 2nd Trimester (`100–250 mm`) | 96 | 97.86% | 1.40 | 2.43 | +1.77 | 2.40 |
| `BaselineUNet (despeckled)` | 3rd Trimester (`>250 mm`) | 27 | 97.39% | 2.84 | 5.39 | +4.13 | 4.80 |

- Literature & trimester comparison chart: `reports/figures/literature_comparison.png`
- Qualitative ellipse overlays & Bland-Altman plots: `reports/figures/postprocessing_ellipse_overlays.png`, `reports/figures/biometric_bland_altman_and_scatter.png`

### 5.3 Key Findings & Error Analysis
1. **Least-squares ellipse fitting improves segmentation overlap on both arms:** Converting cleaned U-Net masks to fitted ellipses improved mean DSC from `96.82% -> 97.12%` (`+0.30%`) on `raw_norm` and `96.30% -> 96.61%` (`+0.31%`) on `despeckled`, confirming that geometric fitting smoothly bridges shadowed skull arcs.
2. **Experimental answer to our speckle-reduction hypothesis:** Aggressive Perona-Malik Anisotropic Diffusion (`EPI = 0.6699`) slightly improved global pixel sensitivity (`97.59% vs. 97.27%`) and outperformed `raw_norm` on several noisy 2nd/3rd-trimester scans (e.g., `895_2HC.png`: `8.68 mm vs. 10.80 mm`; `672_HC.png`: `8.03 mm vs. 10.04 mm`). However, on a faint, small 1st-trimester scan (`10_2HC.png`, `GT HC = 62.31 mm`), over-smoothing erased the thin skull wall, causing a complete miss (`DSC = 0.00%`, `AD = 41.99 mm`) whereas `raw_norm` succeeded (`DSC = 86.53%`, `AD = 2.97 mm`). That single failure drove the large standard deviation (`DSC SD = 8.12%`) and higher overall AD (`3.27 mm`).
3. **Slight positive overestimation bias in 3rd trimester:** Both models exhibit positive signed bias (`MD = +1.00 mm` on `raw_norm`, `+1.74 mm` on `despeckled`), most pronounced on large shadowed 3rd-trimester skulls (`999_HC.png`: `GT = 330.7 mm`, `Pred = 347.0 mm`), partly driven by `pos_weight = 2.269` slightly favoring foreground boundary expansion and coarse `128 × 192` network input resolution.

---

## 6. Limitations & Planned Improvements for Phase 3

### Current Limitations
- **Coarse baseline input resolution (`128 × 192`) and small capacity (`491k` parameters):** Fine skull-wall localization on large 3rd-trimester scans (where `1 px` at `128 × 192` spans `~0.8–1.5 mm`) limits sub-millimeter boundary accuracy.
- **Over-smoothing trade-off of Perona-Malik diffusion:** Lower edge preservation (`EPI = 0.6699`) harmed low-contrast 1st-trimester fetal heads.
- **Slight positive HC bias (`+1.00 mm`):** Training with `pos_weight = 2.269` shifts the `0.5` probability contour slightly outward; threshold tuning on the validation split or boundary-aware losses can calibrate `MD` closer to `0.0 mm`.
- **Single-stage segmentation only:** Unlike Sobhaninia et al. (2019), the baseline does not yet incorporate direct ellipse/boundary regression or multi-task supervision.

### Phase 3 Action Plan
1. **Controlled Despeckling Ablation:** Evaluate higher-EPI filters already benchmarked in `src/preprocessing/despeckle.py` — specifically **Non-Local Means (`EPI = 0.9168`, highest `CNR = 0.5722`)**, **Bilateral (`EPI = 0.8302`)**, and gentler **SRAD / tuned Anisotropic Diffusion** — plus multi-channel input stacking (`[Raw Normalized, Despeckled]`) so the network leverages denoised interiors without losing raw skull-edge sharpness.
2. **Higher-Resolution Upgraded Architecture & Calibration:** Train at full standardized `256 × 384` resolution (using GPU acceleration) with an upgraded backbone (e.g., Attention U-Net / ResNet-encoder U-Net + Boundary/Focal-Tversky loss) and validation-calibrated probability thresholding to eliminate the `+1.0 mm` bias.
3. **Clinical Decision-Support GUI (`src/clinical/`):** Build an interactive clinician-facing desktop/web GUI that accepts an ultrasound image, pixel spacing (`mm/px`), and gestational age (weeks), displays the segmented contour and fitted ellipse with predicted HC (`mm`), and plots the patient's measurement against **Verburg et al. (2008)** 10th, 50th, and 90th percentile fetal head growth reference charts to flag potential microcephaly, macrocephaly, or IUGR.
