import pandas as pd
import plotly.express as px
import streamlit as st

from loaders import load_dataset, load_model
from modeling import (
    add_derived, categories_from_model, expected_features, forest_interval, predict_usd, what_if_drivers,
)
from schema import DERIVED, FIELD_BY_NAME, GROUPS, LABELS, visible_fields
from settings import ACCENT, ALERT, MAX_BATCH_ROWS, MODEL_PATH, NEUTRAL, TARGET
from ui import demo_banner, fmt_num, fmt_usd, missing_state, show, style_fig

st.title("Prediksi Biaya Klaim")
demo_banner()

model, err = load_model()
if model is None:
    missing_state("Model", MODEL_PATH, err)
    st.stop()

features = expected_features(model)
if features is None:
    st.error("Model tidak menyimpan nama fitur. Simpan pipeline lengkap yang di-fit pada DataFrame, bukan estimator saja.")
    st.stop()
unknown = [f for f in features if f not in FIELD_BY_NAME and f != DERIVED]
if unknown:
    st.error(f"Fitur model belum ada di app/schema.py: {', '.join(unknown)}.")
    st.stop()

reference, _ = load_dataset()
if reference is not None and TARGET not in reference.columns:
    reference = None

model_categories = categories_from_model(model)
visible = visible_fields(features)


def options_for(field):
    if field.name in model_categories:
        return model_categories[field.name]
    if reference is not None and field.name in reference.columns:
        return sorted(reference[field.name].dropna().astype(str).unique())
    return list(field.options)


def default_for(field, options):
    if field.kind == "category":
        return field.default if field.default in options else options[0]
    return field.default


baseline_values = {f.name: default_for(f, options_for(f) if f.kind == "category" else ()) for f in visible}


def render(field):
    key = f"in_{field.name}"
    if field.kind == "int":
        return st.number_input(field.label, int(field.low), int(field.high), int(field.default), 1, key=key, help=field.help or None)
    if field.kind == "number":
        return st.number_input(field.label, float(field.low), float(field.high), float(field.default), float(field.step), key=key, help=field.help or None)
    if field.kind == "binary":
        return int(st.checkbox(field.label, key=key))
    options = options_for(field)
    return st.selectbox(field.label, options, index=options.index(default_for(field, options)), key=key)


def problems_in(values):
    found = []
    if "systolic_bp" in values and "diastolic_bp" in values and values["systolic_bp"] <= values["diastolic_bp"]:
        found.append("Tekanan sistolik harus lebih tinggi dari diastolik.")
    if values.get("days_hospitalized_last_3yrs", 0) > 0 and values.get("hospitalizations_last_3yrs", 1) == 0:
        found.append("Hari rawat inap lebih dari 0 tetapi jumlah rawat inap 0. Perbaiki salah satunya.")
    return found


def risk_band(value):
    if reference is None:
        return None
    median, p90 = reference[TARGET].quantile([0.5, 0.9])
    if value < median:
        return "Rendah", median, p90
    return ("Sedang" if value < p90 else "Tinggi"), median, p90


def run_prediction(values):
    row = add_derived(pd.DataFrame([values]))
    base = add_derived(pd.DataFrame([baseline_values]))
    interval = forest_interval(model, row[features])
    return {
        "estimate": float(predict_usd(model, row, features)[0]),
        "interval": None if interval is None else (float(interval[0][0]), float(interval[1][0])),
        "drivers": what_if_drivers(model, row, base, features),
    }


tab_single, tab_batch = st.tabs(["Satu nasabah", "Banyak nasabah (CSV)"])

with tab_single:
    values = {}
    with st.form("form_prediksi"):
        for group in GROUPS:
            fields = [f for f in visible if f.group == group]
            if not fields:
                continue
            st.subheader(group)
            columns = st.columns(3)
            for i, field in enumerate(fields):
                with columns[i % 3]:
                    values[field.name] = render(field)
        submitted = st.form_submit_button("Hitung estimasi biaya", type="primary")

    if submitted:
        issues = problems_in(values)
        st.session_state.pop("hasil", None)
        for issue in issues:
            st.error(issue)
        if not issues:
            try:
                with st.spinner("Menghitung..."):
                    st.session_state["hasil"] = run_prediction(values)
            except Exception as exc:
                st.error(f"Prediksi gagal: {type(exc).__name__}: {exc}")

    result = st.session_state.get("hasil")
    if result is None:
        st.info("Isi form lalu tekan Hitung estimasi biaya.")
    else:
        estimate = result["estimate"]
        st.header("Hasil")
        band = risk_band(estimate)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimasi biaya tahunan", fmt_usd(estimate))
        if result["interval"]:
            lo, hi = result["interval"]
            c2.metric("Sebaran pohon, persentil 10-90", f"{fmt_usd(lo, 0)} - {fmt_usd(hi, 0)}",
                      help="Sebaran prediksi antar pohon Random Forest. Ini bukan interval kepercayaan yang terkalibrasi.")
        if band:
            c3.metric("Kategori biaya", band[0])
            st.caption(
                f"Kategori mengikuti data referensi: Rendah di bawah median ({fmt_usd(band[1], 0)}), "
                f"Sedang sampai persentil 90 ({fmt_usd(band[2], 0)}), Tinggi di atasnya. "
                "Batas ini aturan sederhana aplikasi, bukan standar aktuaria."
            )
        if reference is not None:
            share = (reference[TARGET] <= estimate).mean() * 100
            st.write(f"Estimasi ini lebih tinggi dari {fmt_num(share, 1)}% biaya medis nasabah dalam data referensi.")
            hist = px.histogram(
                reference[reference[TARGET] <= reference[TARGET].quantile(0.99)], x=TARGET, nbins=70,
                color_discrete_sequence=[NEUTRAL], labels={TARGET: "Biaya medis tahunan (USD)"},
            )
            hist.add_vline(x=estimate, line_color=ALERT, line_width=3, annotation_text="Estimasi")
            show(style_fig(hist, 300))

        st.subheader("Faktor yang paling mengubah estimasi")
        drivers = result["drivers"].head(10)
        if drivers.empty:
            st.info("Profil sama dengan profil referensi, sehingga tidak ada faktor pembeda.")
        else:
            plot = drivers.assign(
                nama=drivers["fitur"].map(lambda c: LABELS.get(c, c)),
                arah=drivers["selisih"].map(lambda v: "menaikkan" if v >= 0 else "menurunkan"),
            ).iloc[::-1]
            show(style_fig(px.bar(
                plot, x="selisih", y="nama", color="arah", orientation="h",
                color_discrete_map={"menaikkan": ALERT, "menurunkan": ACCENT},
                labels={"selisih": "Selisih estimasi (USD)", "nama": "", "arah": ""},
            ), max(260, 36 * len(plot))))
            st.caption(
                "Selisih estimasi bila fitur tersebut diganti ke nilai profil referensi, satu per satu. "
                "Ini analisis what-if, bukan nilai Shapley, dan bisa tidak berjumlah sama dengan total."
            )
        with st.expander("Profil referensi"):
            st.dataframe(
                pd.DataFrame({"fitur": [LABELS.get(k, k) for k in baseline_values], "nilai": [str(v) for v in baseline_values.values()]}),
                hide_index=True, width="stretch",
            )
        st.caption("Estimasi pendukung keputusan, bukan keputusan akhir penerimaan polis atau penetapan premi.")

with tab_batch:
    st.write("Unggah CSV berisi kolom fitur yang sama dengan form. Templat di bawah memuat kolom yang dibutuhkan.")
    template = pd.DataFrame([baseline_values])[[f.name for f in visible]]
    st.download_button("Unduh templat CSV", template.to_csv(index=False).encode("utf-8"), "templat_input.csv", "text/csv")
    upload = st.file_uploader("File CSV", type="csv")
    if upload is None:
        st.info("Belum ada file yang diunggah.")
    else:
        try:
            data = add_derived(pd.read_csv(upload), overwrite=False)
        except Exception as exc:
            st.error(f"File tidak bisa dibaca: {exc}")
            st.stop()
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Kolom berikut tidak ada di file: {', '.join(missing)}.")
        elif len(data) > MAX_BATCH_ROWS:
            st.error(f"Maksimal {fmt_num(MAX_BATCH_ROWS, 0)} baris per file. File ini {fmt_num(len(data), 0)} baris.")
        elif data.empty:
            st.warning("File tidak berisi baris data.")
        else:
            try:
                with st.spinner("Menghitung..."):
                    scored = data.assign(prediksi_biaya_usd=predict_usd(model, data, features).round(2))
            except Exception as exc:
                st.error(f"Prediksi gagal: {type(exc).__name__}: {exc}")
                st.stop()
            c1, c2, c3 = st.columns(3)
            c1.metric("Baris diproses", fmt_num(len(scored), 0))
            c2.metric("Rata-rata estimasi", fmt_usd(scored["prediksi_biaya_usd"].mean()))
            c3.metric("Estimasi tertinggi", fmt_usd(scored["prediksi_biaya_usd"].max()))
            st.dataframe(scored.head(200), width="stretch")
            st.download_button("Unduh hasil CSV", scored.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv")
            st.caption("File diproses di memori sesi dan tidak ditulis ke disk oleh aplikasi. Tabel menampilkan 200 baris pertama.")
