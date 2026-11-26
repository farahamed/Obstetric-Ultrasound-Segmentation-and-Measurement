"""Central configuration for paths, dataset columns, and experiment constants."""

from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw HC18 paths
RAW_TRAIN_DIR = RAW_DATA_DIR / "training_set"
RAW_TEST_DIR = RAW_DATA_DIR / "test_set"
RAW_TRAIN_CSV = RAW_DATA_DIR / "training_set_pixel_size_and_HC.csv"
RAW_TEST_CSV = RAW_DATA_DIR / "test_set_pixel_size.csv"

# CSV column names (verified against HC18 metadata files)
COL_FILENAME = "filename"
COL_PIXEL_SIZE = "pixel size"
COL_HC_MM = "head circumference (mm)"

# Reproducibility & split ratios
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default target spatial size for standardized model input (height, width)
# HC18 native majority resolution is 540 x 800 (H x W); 256 x 384 preserves ~1.5 aspect ratio.
DEFAULT_TARGET_SIZE = (256, 384)
