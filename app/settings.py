"""Path artefak dan konstanta yang dipakai semua halaman."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# MEDCOST_ARTIFACTS mengarahkan aplikasi ke folder data uji (lihat scripts/buat_artefak_uji.py).
_override = os.environ.get("MEDCOST_ARTIFACTS")
DEMO_MODE = bool(_override)
ARTIFACT_ROOT = (ROOT / _override).resolve() if _override else ROOT

MODEL_PATH = ARTIFACT_ROOT / "models" / "model.joblib"
DATA_DIR = ARTIFACT_ROOT / "data"
DATA_FILENAME = "medical_insurance.csv"
TABLES_DIR = ARTIFACT_ROOT / "reports" / "tables"
METRICS_PATH = TABLES_DIR / "metrics.csv"
PREDICTIONS_PATH = TABLES_DIR / "predictions_test.csv"
IMPORTANCE_PATH = TABLES_DIR / "feature_importance.csv"

TARGET = "annual_medical_cost"
R2_TARGET = 0.75
MAX_BATCH_ROWS = 50_000

ACCENT = "#0F6E6E"
NEUTRAL = "#5B6B7A"
ALERT = "#B5502D"