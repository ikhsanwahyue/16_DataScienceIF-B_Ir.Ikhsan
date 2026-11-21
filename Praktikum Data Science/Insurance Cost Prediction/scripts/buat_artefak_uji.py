"""Membuat artefak SINTETIS di folder .dummy/ agar aplikasi Streamlit bisa dicoba sebelum model asli jadi.

Data dan angka dari skrip ini bukan hasil proyek. Jalankan aplikasi dengan:
    MEDCOST_ARTIFACTS=.dummy streamlit run app/streamlit_app.py
(Windows PowerShell: $env:MEDCOST_ARTIFACTS=".dummy"; streamlit run app/streamlit_app.py)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = Path(__file__).resolve().parents[1] / ".dummy"
N = 8000
rng = np.random.default_rng(42)

DISEASES = ["hypertension", "diabetes", "copd", "cardiovascular_disease", "cancer_history",
            "kidney_disease", "liver_disease", "arthritis", "mental_health", "asthma"]
PREVALENCE = [0.2, 0.09, 0.035, 0.05, 0.02, 0.015, 0.01, 0.1, 0.13, 0.06]
PROCS = ["proc_imaging_count", "proc_surgery_count", "proc_physio_count", "proc_consult_count", "proc_lab_count"]

df = pd.DataFrame({
    "age": rng.integers(18, 91, N),
    "sex": rng.choice(["Female", "Male", "Other"], N, p=[0.49, 0.49, 0.02]),
    "smoker": rng.choice(["Never", "Former", "Current"], N, p=[0.7, 0.18, 0.12]),
    "alcohol_freq": rng.choice(["None", "Occasional", "Weekly", "Daily"], N, p=[0.2, 0.45, 0.25, 0.1]),
    "bmi": rng.normal(27, 5, N).clip(14, 55).round(1),
    "systolic_bp": rng.normal(122, 14, N).clip(85, 210).round(0),
    "ldl": rng.normal(110, 28, N).clip(40, 250).round(1),
    "hba1c": rng.normal(5.6, 0.7, N).clip(4, 12).round(2),
})
df["diastolic_bp"] = (df["systolic_bp"] - rng.normal(45, 6, N)).clip(50, 130).round(0)
df.loc[rng.random(N) < 0.3, "alcohol_freq"] = np.nan
for name, p in zip(DISEASES, PREVALENCE):
    df[name] = (rng.random(N) < p).astype(int)
df["chronic_count"] = df[DISEASES].sum(axis=1)
df["hospitalizations_last_3yrs"] = rng.poisson(0.3 + 0.4 * df["chronic_count"])
df["days_hospitalized_last_3yrs"] = df["hospitalizations_last_3yrs"] * rng.integers(1, 6, N)
df["visits_last_year"] = rng.poisson(2 + df["chronic_count"])
df["medication_count"] = rng.poisson(0.5 + 1.2 * df["chronic_count"])
for name in PROCS:
    df[name] = rng.poisson(0.5 + 0.3 * df["chronic_count"])
df["had_major_procedure"] = (df["hospitalizations_last_3yrs"] + df["proc_surgery_count"] > 1).astype(int)

signal = (
    6.8 + 0.01 * df["age"] + 0.35 * (df["smoker"] == "Current") + 0.12 * (df["smoker"] == "Former")
    + 0.18 * df["chronic_count"] + 0.09 * df["hospitalizations_last_3yrs"] + 0.03 * df["visits_last_year"]
    + 0.006 * (df["bmi"] - 25).clip(0) + 0.3 * df["had_major_procedure"]
)
df["annual_medical_cost"] = np.exp(signal - 1.2 + rng.normal(0, 0.45, N)).round(2)

features = [c for c in df.columns if c != "annual_medical_cost"]
categorical = ["sex", "smoker", "alcohol_freq"]
numeric = [c for c in features if c not in categorical]

def make(estimator):
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return TransformedTargetRegressor(Pipeline([("prep", prep), ("model", estimator)]), func=np.log1p, inverse_func=np.expm1)

X_train, X_test, y_train, y_test = train_test_split(df[features], df["annual_medical_cost"], test_size=0.2, random_state=42)
candidates = {
    "Dummy (rata-rata)": make(DummyRegressor()),
    "Linear Regression": make(LinearRegression()),
    "Random Forest": make(RandomForestRegressor(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=42)),
}
rows, fitted = [], {}
for name, est in candidates.items():
    est.fit(X_train, y_train)
    fitted[name] = est
    tr, te = est.predict(X_train), est.predict(X_test)
    rows.append({
        "model": name, "r2_train": r2_score(y_train, tr), "r2_test": r2_score(y_test, te),
        "rmse_train": mean_squared_error(y_train, tr) ** 0.5, "rmse_test": mean_squared_error(y_test, te) ** 0.5,
        "mae_train": mean_absolute_error(y_train, tr), "mae_test": mean_absolute_error(y_test, te),
    })
metrics = pd.DataFrame(rows)
best = fitted["Random Forest"]

for sub in ("data/raw", "models", "reports/tables"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "data/raw/medical_insurance.csv", index=False)
joblib.dump(best, OUT / "models/model.joblib")
metrics.to_csv(OUT / "reports/tables/metrics.csv", index=False)
pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": best.predict(X_test), "sex": X_test["sex"].to_numpy(),
              "age": X_test["age"].to_numpy()}).to_csv(OUT / "reports/tables/predictions_test.csv", index=False)
forest = best.regressor_
names = forest.named_steps["prep"].get_feature_names_out()
pd.DataFrame({"feature": names, "importance": forest.named_steps["model"].feature_importances_}).to_csv(
    OUT / "reports/tables/feature_importance.csv", index=False)
print(metrics.round(3).to_string(index=False))
print("Artefak uji ditulis ke", OUT)