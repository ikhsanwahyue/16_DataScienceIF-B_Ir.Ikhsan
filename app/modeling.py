"""Fungsi inferensi dan analisis yang tidak bergantung pada Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from schema import CHRONIC_SOURCES, DERIVED


def inner_pipeline(model):
    """Buka TransformedTargetRegressor bila ada, sisanya dikembalikan apa adanya."""
    return getattr(model, "regressor_", model)


def expected_features(model):
    names = getattr(inner_pipeline(model), "feature_names_in_", None)
    return list(names) if names is not None else None


def add_derived(df, overwrite=True):
    out = df.copy()
    sources = list(CHRONIC_SOURCES)
    if all(c in out.columns for c in sources) and (overwrite or DERIVED not in out.columns):
        out[DERIVED] = out[sources].sum(axis=1)
    return out


def predict_usd(model, df, features):
    return np.clip(np.asarray(model.predict(df[features]), dtype=float), 0, None)


def categories_from_model(model):
    """Kategori yang dikenal encoder di dalam pipeline, supaya form tidak mengirim nilai asing."""
    found = {}
    steps = getattr(inner_pipeline(model), "steps", [])
    transformer = next((s for _, s in steps if hasattr(s, "transformers_")), None)
    if transformer is None:
        return found
    for _, part, cols in transformer.transformers_:
        encoder = part.steps[-1][1] if hasattr(part, "steps") else part
        cats = getattr(encoder, "categories_", None)
        if cats is None or not isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            continue
        for col, values in zip(cols, cats):
            found[col] = [str(v) for v in values if pd.notna(v)]
    return found


def forest_interval(model, X, low=10, high=90):
    """Persentil sebaran prediksi antar pohon. Hanya untuk Random Forest/Extra Trees; selain itu None."""
    try:
        ttr = model if hasattr(model, "regressor_") else None
        pipe = inner_pipeline(model)
        estimator = pipe.steps[-1][1]
        if not isinstance(estimator, (RandomForestRegressor, ExtraTreesRegressor)):
            return None
        Z = pipe[:-1].transform(X)
        per_tree = np.column_stack([t.predict(Z) for t in estimator.estimators_])
        if ttr is not None:
            flat = ttr.transformer_.inverse_transform(per_tree.reshape(-1, 1))
            per_tree = flat.reshape(per_tree.shape)
        lo, hi = np.percentile(per_tree, [low, high], axis=1)
        return np.clip(lo, 0, None), np.clip(hi, 0, None)
    except Exception:
        return None


def what_if_drivers(model, row, baseline, features):
    """Selisih estimasi bila tiap fitur yang berbeda dikembalikan ke nilai profil referensi."""
    full = predict_usd(model, row, features)[0]
    names, variants = [], []
    for f in features:
        if row.at[0, f] == baseline.at[0, f]:
            continue
        alt = row.copy()
        alt.at[0, f] = baseline.at[0, f]
        names.append(f)
        variants.append(add_derived(alt))
    if not names:
        return pd.DataFrame(columns=["fitur", "selisih"])
    preds = predict_usd(model, pd.concat(variants, ignore_index=True), features)
    out = pd.DataFrame({"fitur": names, "selisih": full - preds})
    return out.reindex(out["selisih"].abs().sort_values(ascending=False).index)


def psi(reference, current, bins=10):
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > 10:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    else:
        ref, cur = ref.astype(str), cur.astype(str)
        cats = sorted(set(ref) | set(cur))
        p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        q = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_status(value):
    if np.isnan(value):
        return "tidak dihitung"
    if value < 0.1:
        return "stabil"
    return "perlu dipantau" if value < 0.25 else "bergeser"


def drift_table(reference, current, columns):
    rows = []
    for col in columns:
        value = psi(reference[col], current[col])
        ks_stat = ks_p = np.nan
        if pd.api.types.is_numeric_dtype(reference[col]) and reference[col].nunique() > 10:
            ks_stat, ks_p = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append({"fitur": col, "PSI": value, "KS": ks_stat, "p-value KS": ks_p, "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False, ignore_index=True)


def error_summary(df, group_col, min_n=30):
    """Ringkasan error per kelompok dan p-value Kruskal-Wallis pada error absolut."""
    work = df.assign(_err=df["y_pred"] - df["y_true"]).dropna(subset=[group_col])
    work["_abs"] = work["_err"].abs()
    rows, samples = [], []
    for name, part in work.groupby(group_col, observed=True):
        if len(part) < min_n:
            continue
        rows.append({
            "kelompok": str(name), "n": len(part), "MAE": part["_abs"].mean(),
            "RMSE": float(np.sqrt((part["_err"] ** 2).mean())), "bias": part["_err"].mean(),
        })
        samples.append(part["_abs"].to_numpy())
    p_value = float(stats.kruskal(*samples).pvalue) if len(samples) >= 2 else None
    return pd.DataFrame(rows), p_value


def regression_scores(y_true, y_pred):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    y = np.asarray(y_true, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot else float("nan"),
        "bias": float(err.mean()),
    }