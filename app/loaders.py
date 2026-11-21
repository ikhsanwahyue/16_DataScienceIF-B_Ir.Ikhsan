"""Pemuat artefak dengan cache. Setiap fungsi mengembalikan (objek, pesan_error)."""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from schema import DERIVED, FIELDS
from settings import DATA_PATH, MODEL_PATH, TARGET

DATASET_COLUMNS = frozenset([TARGET, DERIVED] + [f.name for f in FIELDS])


def _stamp(path):
    return path.stat().st_mtime if path.exists() else None


@st.cache_resource(show_spinner="Memuat model...")
def _load_bundle(path_str, stamp):
    obj = joblib.load(path_str)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    if hasattr(obj, "predict"):
        return {"model": obj}
    raise ValueError("isi file bukan bundle dari main.ipynb (dict dengan kunci 'model')")


@st.cache_data(show_spinner="Membaca data...")
def _read_csv(path_str, stamp):
    return pd.read_csv(path_str, usecols=lambda c: c in DATASET_COLUMNS)


def load_bundle():
    if not MODEL_PATH.exists():
        return None, None
    try:
        return _load_bundle(str(MODEL_PATH), _stamp(MODEL_PATH)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_model():
    bundle, err = load_bundle()
    return (None if bundle is None else bundle["model"]), err


def load_artifact(key):
    """Tabel hasil evaluasi di dalam bundle (metrics, predictions_test, feature_importance)."""
    bundle, err = load_bundle()
    if bundle is None:
        return None, err
    return bundle.get(key), None


def dataset_path():
    return DATA_PATH


def load_dataset():
    """Hanya kolom klinis dan target yang dimuat; identitas dan variabel sosio-ekonomi tidak dibaca."""
    if not DATA_PATH.exists():
        return None, None
    try:
        return _read_csv(str(DATA_PATH), _stamp(DATA_PATH)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"