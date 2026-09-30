"""Application configuration and constants shared across modules."""

from pathlib import Path

APP_NAME = "Smart Data Quality Analyzer & Automated EDA Tool"
APP_VERSION = "1.0.0"

# Maximum rows loaded into memory for analysis (bright limits for very large files).
MAX_ROWS = 200_000
# Chunk size used for lazy/chunked CSV loading.
CHUNK_SIZE = 50_000

# Numeric thresholds used by the quality dimension calculations.
VALID_EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
VALID_TEL_PATTERN = r"^\+?\d[\d\s().-]{6,}$"

# Outlier detection defaults.
OUTLIER_IQR_DEFAULT = 1.5
OUTLIER_Z_MIN_ROWS = 8

# Rendering tweaks.
PLOTLY_TEMPLATE = "plotly_white"
CATEGORY_MAX_UNIQUE_DISPLAY = 15
SKEW_THRESHOLD = 1.0

# Where generated reports/temp artifacts live.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLEANING_METHODS_LABELS = {
    "drop": "Drop rows with missing values",
    "mean": "Impute with column mean",
    "median": "Impute with column median",
    "mode": "Impute with column mode",
    "ffill": "Forward fill",
    "bfill": "Backward fill",
    "constant": "Impute with a fixed value",
}

OUTLIER_ACTIONS_LABELS = {
    "flag": "Flag (add outlier column, keep data)",
    "cap": "Cap/Winsorize to bounds",
    "remove": "Remove outlier rows",
}