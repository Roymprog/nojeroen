import os
from pathlib import Path

# Project root: src/whospeaks/config.py → ../../.. = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths — env vars override; defaults are relative to the project root
DATA_DIR = os.environ.get(
    "WHOSPEAKS_DATA_DIR", str(_PROJECT_ROOT / "data" / "labeled")
)
MODEL_DIR = os.environ.get(
    "WHOSPEAKS_MODEL_DIR", str(_PROJECT_ROOT / "models")
)

# Session definitions
SESSIONS = {
    "S1": "2025-08-03-rinkeldekinkel_0_to_3600_minutes",
    "S2": "2025-08-03-rinkeldekinkel_3600_to_7200_minutes",
    "S3": "2025-08-03-rinkeldekinkel_7200_to_10800_minutes",
    "S4": "2025-11-24-van-inkels-choice_0_to_3600_minutes",
}

# Split assignment
TRAIN_SESSIONS = ["S1", "S2"]
VAL_SESSION = "S3"
TEST_SESSION = "S4"

# Audio parameters
SAMPLE_RATE = 16000
WINDOW_SIZE_S = 2.0
STRIDE_S = 1.0
MIN_TAIL_S = 1.5

# Feature parameters
EMBEDDING_DIM = 256
FEATURE_TYPE = "resemblyzer_ge2e"

# Labels
POSITIVE_LABEL = "JEROEN_VAN_INKEL"
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

# Model filenames
MODEL_FILENAME = "model.joblib"
CONFIG_FILENAME = "config.json"

# Threshold tuning
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
TARGET_PRECISION = 0.95
