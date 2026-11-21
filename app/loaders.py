"""Pemuat artefak dengan cache. Setiap fungsi mengembalikan (objek, pesan_error)."""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from schema import DERIVED, FIELDS
from settings import DATA_DIR, DATA_FILENAME, MODEL_PATH, TARGET

DATASET_COLUMNS = frozenset([TARGET, DERIVED] + [f.name for f in FIELDS])


def _stamp(path):
    return path.stat().st_mtime if path.exists() else None


@st.cache_resource(show_spinner="Memuat model...")
def _load_model(path_str, stamp):
    return joblib.load(path_str)


@st.cache_data(show_spinner="Membaca data...")
def _read_csv(path_str, stamp, only_dataset_columns):
    usecols = (lambda c: c in DATASET_COLUMNS) if only_dataset_columns else None
    return pd.read_csv(path_str, usecols=usecols)


def load_model():
    if not MODEL_PATH.exists():
        return None, None
    try:
        return _load_model(str(MODEL_PATH), _stamp(MODEL_PATH)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_table(path, only_dataset_columns=False):
    if not path.exists():
        return None, None
    try:
        return _read_csv(str(path), _stamp(path), only_dataset_columns), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def dataset_path():
    preferred = DATA_DIR / DATA_FILENAME
    if preferred.exists() or not DATA_DIR.exists():
        return preferred
    csvs = sorted(DATA_DIR.glob("*.csv"))
    return csvs[0] if csvs else preferred


def load_dataset():
    """Hanya kolom klinis dan target yang dimuat; identitas dan variabel sosio-ekonomi tidak dibaca."""
    return load_table(dataset_path(), only_dataset_columns=True)