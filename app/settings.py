"""Path artefak dan konstanta yang dipakai semua halaman."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"
DATA_PATH = ROOT / "data" / "medical_insurance.csv"

TARGET = "annual_medical_cost"
R2_TARGET = 0.75
MAX_BATCH_ROWS = 50_000

ACCENT = "#0F6E6E"
NEUTRAL = "#5B6B7A"
ALERT = "#B5502D"