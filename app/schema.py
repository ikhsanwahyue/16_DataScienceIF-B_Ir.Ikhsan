"""Skema form input. Nama kolom harus sama dengan kolom yang dipakai saat melatih model."""
from __future__ import annotations

from dataclasses import dataclass

DERIVED = "chronic_count"

G_PROFILE = "Profil dan gaya hidup"
G_VITALS = "Pemeriksaan klinis"
G_DISEASE = "Riwayat diagnosis"
G_USE = "Pemanfaatan fasilitas kesehatan"
GROUPS = (G_PROFILE, G_VITALS, G_DISEASE, G_USE)


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str  # "int", "number", "binary", "category"
    default: object
    group: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    options: tuple = ()
    help: str = ""


def _int(name, label, default, low, high, group, help=""):
    return Field(name, label, "int", default, group, low, high, 1, help=help)


def _num(name, label, default, low, high, step, group, help=""):
    return Field(name, label, "number", default, group, low, high, step, help=help)


def _flag(name, label, group=G_DISEASE):
    return Field(name, label, "binary", 0, group)


def _cat(name, label, default, options, group=G_PROFILE):
    return Field(name, label, "category", default, group, options=options)


FIELDS = (
    _int("age", "Usia (tahun)", 35, 18, 90, G_PROFILE),
    _cat("sex", "Jenis kelamin", "Female", ("Female", "Male", "Other")),
    _cat("smoker", "Status merokok", "Never", ("Never", "Former", "Current")),
    _cat("alcohol_freq", "Frekuensi alkohol", "Unknown", ("Unknown", "Occasional", "Weekly", "Daily")),
    _num("bmi", "BMI (kg/m²)", 24.0, 10.0, 60.0, 0.1, G_VITALS),
    _num("systolic_bp", "Tekanan sistolik (mmHg)", 120.0, 70.0, 250.0, 1.0, G_VITALS),
    _num("diastolic_bp", "Tekanan diastolik (mmHg)", 78.0, 40.0, 150.0, 1.0, G_VITALS),
    _num("ldl", "Kolesterol LDL (mg/dL)", 100.0, 20.0, 400.0, 1.0, G_VITALS),
    _num("hba1c", "HbA1c (%)", 5.4, 3.0, 18.0, 0.1, G_VITALS),
    _flag("hypertension", "Hipertensi"),
    _flag("diabetes", "Diabetes"),
    _flag("copd", "PPOK (COPD)"),
    _flag("cardiovascular_disease", "Penyakit kardiovaskular"),
    _flag("cancer_history", "Riwayat kanker"),
    _flag("kidney_disease", "Penyakit ginjal"),
    _flag("liver_disease", "Penyakit hati"),
    _flag("arthritis", "Arthritis"),
    _flag("mental_health", "Diagnosis kesehatan mental"),
    _flag("asthma", "Asma"),
    _int("visits_last_year", "Kunjungan ke penyedia layanan, 1 tahun terakhir", 2, 0, 100, G_USE),
    _int("hospitalizations_last_3yrs", "Rawat inap, 3 tahun terakhir", 0, 0, 30, G_USE),
    _int("days_hospitalized_last_3yrs", "Total hari rawat inap, 3 tahun terakhir", 0, 0, 365, G_USE),
    _int("medication_count", "Jumlah obat aktif", 0, 0, 40, G_USE),
    _int("proc_imaging_count", "Prosedur pencitraan (X-ray, MRI, CT)", 0, 0, 30, G_USE),
    _int("proc_surgery_count", "Jumlah operasi", 0, 0, 20, G_USE),
    _int("proc_physio_count", "Sesi fisioterapi/psikologis", 0, 0, 100, G_USE),
    _int("proc_consult_count", "Konsultasi spesialis", 0, 0, 50, G_USE),
    _int("proc_lab_count", "Tes laboratorium", 1, 0, 100, G_USE),
    _flag("had_major_procedure", "Kejadian medis besar dalam 1 tahun terakhir", G_USE),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
NUMERIC_NAMES = [f.name for f in FIELDS if f.kind in ("int", "number")]
BINARY_NAMES = [f.name for f in FIELDS if f.kind == "binary"]
CATEGORY_NAMES = [f.name for f in FIELDS if f.kind == "category"]
LABELS = {f.name: f.label for f in FIELDS} | {DERIVED: "Jumlah penyakit kronis"}

# Harus sama dengan cara chronic_count dibentuk di notebook; cek dengan df[sumber].sum(axis=1).eq(df.chronic_count).mean().
CHRONIC_SOURCES = (
    "hypertension", "diabetes", "copd", "cardiovascular_disease", "cancer_history",
    "kidney_disease", "liver_disease", "arthritis", "mental_health", "asthma",
)


def visible_fields(features):
    """Field yang tampil di form: fitur model, ditambah sumber chronic_count bila fitur itu dipakai."""
    names = set(features)
    if DERIVED in names:
        names |= set(CHRONIC_SOURCES)
    return [f for f in FIELDS if f.name in names]