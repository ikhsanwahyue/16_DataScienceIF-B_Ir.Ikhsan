import pandas as pd
import plotly.express as px
import streamlit as st

from loaders import dataset_path, load_artifact, load_dataset, load_model
from modeling import add_derived, drift_table, error_summary, expected_features, predict_usd, regression_scores
from schema import DERIVED, FIELDS
from settings import ACCENT, ALERT, MAX_BATCH_ROWS, MODEL_PATH, NEUTRAL, TARGET
from ui import fmt_num, fmt_usd, missing_artifact, missing_state, show, style_fig

st.title("Fairness dan Monitoring")

tab_fair, tab_drift = st.tabs(["Keadilan model", "Monitoring data baru"])

with tab_fair:
    preds, err = load_artifact("predictions_test")
    if preds is None:
        missing_artifact("Prediksi data uji", "predictions_test", err)
    elif not {"y_true", "y_pred"} <= set(preds.columns):
        st.error("File prediksi harus punya kolom y_true dan y_pred.")
    elif not {"sex", "age"} & set(preds.columns):
        st.info("File prediksi perlu kolom sex dan/atau age (nilai asli data uji) untuk menghitung error per kelompok.")
    else:
        if "age" in preds.columns:
            preds = preds.assign(kelompok_usia=pd.cut(
                preds["age"], [0, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"]))
        for column, title in (("sex", "Jenis kelamin"), ("kelompok_usia", "Kelompok usia")):
            if column not in preds.columns:
                continue
            st.subheader(title)
            table, p_value = error_summary(preds, column)
            if table.empty:
                st.warning("Tidak ada kelompok dengan sedikitnya 30 baris.")
                continue
            show(style_fig(px.bar(
                table, x="kelompok", y="MAE", color_discrete_sequence=[ACCENT],
                labels={"kelompok": "", "MAE": "MAE (USD)"},
            ), 300))
            st.dataframe(table, hide_index=True, width="stretch")
            gap = table["MAE"].max() - table["MAE"].min()
            st.write(f"Selisih MAE terbesar antar kelompok: {fmt_usd(gap)} (rasio {fmt_num(table['MAE'].max() / table['MAE'].min(), 2)}).")
            if p_value is not None:
                verdict = "berbeda signifikan" if p_value < 0.05 else "tidak berbeda signifikan"
                st.write(f"Uji Kruskal-Wallis pada error absolut: p = {fmt_num(p_value, 4)}, antar kelompok {verdict} pada taraf 5%.")
        st.caption(
            "Pada data besar, selisih kecil pun bisa signifikan secara statistik, jadi baca p-value bersama besar selisih MAE. "
            "Kelompok dengan kurang dari 30 baris tidak ditampilkan."
        )

with tab_drift:
    reference, err = load_dataset()
    if reference is None:
        missing_state("Dataset referensi", dataset_path(), err)
        st.stop()
    st.write(
        "Unggah CSV data baru untuk membandingkan sebaran fitur dengan data latih. Bila file memuat kolom "
        f"`{TARGET}`, error model pada data baru ikut dihitung."
    )
    upload = st.file_uploader("CSV data baru", type="csv", key="drift_upload")
    if upload is None:
        st.info("Belum ada file yang diunggah.")
        st.stop()
    try:
        new = add_derived(pd.read_csv(upload), overwrite=False)
    except Exception as exc:
        st.error(f"File tidak bisa dibaca: {exc}")
        st.stop()
    if new.empty:
        st.warning("File tidak berisi baris data.")
        st.stop()
    if len(new) > MAX_BATCH_ROWS:
        st.error(f"Maksimal {fmt_num(MAX_BATCH_ROWS, 0)} baris per file.")
        st.stop()

    shared = [c for c in [f.name for f in FIELDS] + [DERIVED] if c in reference.columns and c in new.columns]
    if not shared:
        st.error("Tidak ada kolom fitur yang sama antara file dan data referensi.")
        st.stop()
    drift = drift_table(reference, new, shared)
    show(style_fig(px.bar(
        drift.iloc[::-1], x="PSI", y="fitur", orientation="h", color="status",
        color_discrete_map={"stabil": ACCENT, "perlu dipantau": NEUTRAL, "bergeser": ALERT, "tidak dihitung": "#8FA3B3"},
        labels={"fitur": "", "status": ""},
    ), max(320, 24 * len(drift))))
    st.dataframe(drift, hide_index=True, width="stretch")
    st.caption("Acuan umum PSI: di bawah 0,1 stabil, 0,1 sampai 0,25 perlu dipantau, di atas 0,25 bergeser. KS hanya untuk fitur numerik kontinu.")

    st.subheader("Error model pada data baru")
    model, model_err = load_model()
    if TARGET not in new.columns:
        st.info(f"File tidak memuat kolom `{TARGET}`, sehingga error tidak dapat dihitung.")
    elif model is None:
        missing_state("Model", MODEL_PATH, model_err)
    else:
        features = expected_features(model) or []
        absent = [f for f in features if f not in new.columns]
        if not features or absent:
            st.error(f"Kolom fitur tidak lengkap untuk model: {', '.join(absent) or 'nama fitur model tidak tersimpan'}.")
        else:
            try:
                scores = regression_scores(new[TARGET], predict_usd(model, new, features))
            except Exception as exc:
                st.error(f"Prediksi gagal: {type(exc).__name__}: {exc}")
                st.stop()
            metrics, _ = load_artifact("metrics")
            test_mae = None
            if metrics is not None and {"r2_test", "mae_test"} <= set(metrics.columns):
                test_mae = float(metrics.loc[metrics["r2_test"].idxmax(), "mae_test"])
            c1, c2, c3 = st.columns(3)
            delta = None if test_mae is None else f"{fmt_usd(scores['MAE'] - test_mae)} dibanding data uji"
            c1.metric("MAE data baru", fmt_usd(scores["MAE"]), delta, delta_color="inverse")
            c2.metric("RMSE data baru", fmt_usd(scores["RMSE"]))
            c3.metric("R² data baru", fmt_num(scores["R2"], 3))
            st.caption("MAE yang naik jauh dari data uji, terutama bersama PSI tinggi, menjadi sinyal untuk retraining.")