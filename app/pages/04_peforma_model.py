import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loaders import load_artifact
from modeling import regression_scores
from settings import ACCENT, ALERT, NEUTRAL, R2_TARGET
from ui import fmt_num, fmt_usd, missing_artifact, pretty_name, show, style_fig

st.title("Performa Model")

metrics, err = load_artifact("metrics")
st.header("Perbandingan model")
needed = {"model", "r2_test", "rmse_test", "mae_test"}
if metrics is None:
    missing_artifact("Tabel metrik model", "metrics", err)
elif not needed <= set(metrics.columns):
    st.error(f"Kolom metrik kurang: {', '.join(sorted(needed - set(metrics.columns)))}.")
else:
    table = metrics.copy()
    if "r2_train" in table.columns:
        table["selisih_r2_train_test"] = table["r2_train"] - table["r2_test"]
    st.dataframe(table.sort_values("r2_test", ascending=False), hide_index=True, width="stretch")
    r2_cols = [c for c in ("r2_train", "r2_cv", "r2_test") if c in table.columns]
    long = table.melt("model", r2_cols, var_name="data", value_name="R²")
    show(style_fig(px.bar(
        long, x="model", y="R²", color="data", barmode="group",
        color_discrete_map={"r2_train": NEUTRAL, "r2_cv": "#8FA3B3", "r2_test": ACCENT},
        labels={"model": "", "data": ""},
    ), 340))
    best = table.loc[table["r2_test"].idxmax()]
    reached = best["r2_test"] > R2_TARGET
    (st.success if reached else st.warning)(
        f"R² uji terbaik {fmt_num(best['r2_test'], 3)} ({best['model']}), "
        f"{'di atas' if reached else 'belum melewati'} target {fmt_num(R2_TARGET)}."
    )
    if "selisih_r2_train_test" in table.columns:
        st.caption("Selisih R² latih dan uji yang besar menandakan overfitting; kriteria proyek menghendaki selisih mendekati 0.")

st.header("Kualitas prediksi pada data uji")
preds, err = load_artifact("predictions_test")
if preds is None:
    missing_artifact("Prediksi data uji", "predictions_test", err)
elif not {"y_true", "y_pred"} <= set(preds.columns):
    st.error("File prediksi harus punya kolom y_true dan y_pred.")
elif preds.empty:
    st.warning("File prediksi tidak berisi baris.")
else:
    preds = preds.assign(residual=preds["y_pred"] - preds["y_true"])
    sample = preds.sample(min(len(preds), 5000), random_state=0)
    log_axes = st.toggle("Sumbu logaritmik pada grafik prediksi", value=False)
    left, right = st.columns(2)
    with left:
        top = float(max(sample["y_true"].max(), sample["y_pred"].max()))
        fig = px.scatter(
            sample, x="y_true", y="y_pred", opacity=0.35, color_discrete_sequence=[ACCENT],
            log_x=log_axes, log_y=log_axes, labels={"y_true": "Biaya aktual (USD)", "y_pred": "Biaya prediksi (USD)"},
        )
        fig.add_trace(go.Scatter(x=[1, top], y=[1, top], mode="lines", line=dict(color=ALERT, dash="dash"), name="prediksi = aktual"))
        show(style_fig(fig, 380))
    with right:
        show(style_fig(px.scatter(
            sample, x="y_pred", y="residual", opacity=0.35, color_discrete_sequence=[NEUTRAL],
            labels={"y_pred": "Biaya prediksi (USD)", "residual": "Residual, prediksi dikurangi aktual (USD)"},
        ), 380))
    show(style_fig(px.histogram(
        preds, x="residual", nbins=80, color_discrete_sequence=[ACCENT],
        labels={"residual": "Residual (USD)"},
    ), 300))

    st.subheader("Segmen biaya tinggi")
    percentile = st.slider("Persentil biaya aktual sebagai batas segmen", 50, 99, 90)
    cutoff = float(np.percentile(preds["y_true"], percentile))
    rows = []
    for name, part in (("Biaya tinggi", preds[preds["y_true"] >= cutoff]), ("Lainnya", preds[preds["y_true"] < cutoff])):
        if part.empty:
            continue
        s = regression_scores(part["y_true"], part["y_pred"])
        rows.append({"segmen": name, "n": len(part), "MAE": s["MAE"], "RMSE": s["RMSE"], "bias": s["bias"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Batas segmen {fmt_usd(cutoff)}. Bias negatif berarti prediksi rata-rata lebih rendah dari biaya aktual, "
        "yaitu risiko under-pricing pada nasabah berbiaya tinggi."
    )

st.header("Feature importance")
imp, err = load_artifact("feature_importance")
if imp is None:
    missing_artifact("Tabel feature importance", "feature_importance", err)
elif not {"feature", "importance"} <= set(imp.columns):
    st.error("File importance harus punya kolom feature dan importance.")
elif imp.empty:
    st.warning("File importance tidak berisi baris.")
else:
    imp = imp.sort_values("importance", ascending=False)
    top_n = st.slider("Jumlah fitur", 1, len(imp), min(15, len(imp))) if len(imp) > 1 else 1
    chosen = imp.head(top_n).iloc[::-1].assign(nama=lambda d: d["feature"].map(pretty_name))
    show(style_fig(px.bar(
        chosen, x="importance", y="nama", orientation="h", color_discrete_sequence=[ACCENT],
        labels={"importance": "Importance", "nama": ""},
    ), max(300, 26 * len(chosen))))
    st.caption("Importance menunjukkan seberapa besar fitur dipakai model, bukan hubungan sebab-akibat dengan biaya.")