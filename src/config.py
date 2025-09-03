from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATA_RAW = BASE / "data" / "raw"
DATA_INTERIM = BASE / "data" / "interim"
DATA_PROCESSED = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
FEATURE_COLUMNS = [
# Keep these generic so it works for both UNSW and sim logs
"bytes_out", "bytes_in", "duration_s",
# engineered features that may be created later
"hour", "is_private_ip"
]
TARGET_COLUMN = "label" # used only if available (supervised eval)
# Inference threshold controls aggressiveness of alerts (lower → more alerts)
ANOMALY_THRESHOLD = 0.65 # threshold on scaled anomaly score [0,1]
