import pandas as pd
import plotly.express as px
import streamlit as st

from loaders import dataset_path, load_dataset
from schema import BINARY_NAMES, CATEGORY_NAMES, DERIVED, LABELS, NUMERIC_NAMES
from settings import ACCENT, ALERT, NEUTRAL, TARGET
from ui import fmt_num, fmt_usd, missing_state, show, style_fig

st.title("Eksplorasi Data")

df, err = load_dataset()
if df is None:
    missing_state("Dataset", dataset_path(), err)
    st.stop()
if TARGET not in df.columns:
    st.error(f"Kolom target `{TARGET}` tidak ada di dataset.")
    st.stop()

mask = pd.Series(True, index=df.index)
with st.sidebar:
    st.subheader("Filter data")
    if "age" in df.columns:
        lo, hi = int(df["age"].min()), int(df["age"].max())
        mask &= df["age"].between(*st.slider("Usia", lo, hi, (lo, hi)))
    for col, label in (("sex", "Jenis kelamin"), ("smoker", "Status merokok")):
        if col in df.columns:
            options = sorted(df[col].dropna().astype(str).unique())
            chosen = st.multiselect(label, options, default=options)
            mask &= df[col].astype(str).isin(chosen)

view = df[mask]
if view.empty:
    st.warning("Tidak ada nasabah yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Nasabah terpilih", fmt_num(len(view), 0))
c2.metric("Rata-rata biaya", fmt_usd(view[TARGET].mean()))
c3.metric("Median biaya", fmt_usd(view[TARGET].median()))

tab_target, tab_num, tab_cat, tab_corr, tab_raw = st.tabs(
    ["Biaya medis", "Fitur numerik", "Fitur kategorikal", "Korelasi", "Cuplikan data"]
)

with tab_target:
    log_x = st.toggle("Sumbu X logaritmik", value=False)
    show(style_fig(px.histogram(
        view, x=TARGET, nbins=80, log_x=log_x, color_discrete_sequence=[ACCENT],
        labels={TARGET: "Biaya medis tahunan (USD)"},
    )))
    summary = pd.DataFrame({
        "statistik": ["rata-rata", "median", "simpangan baku", "persentil 90", "persentil 99", "maksimum", "skewness"],
        "nilai": [
            view[TARGET].mean(), view[TARGET].median(), view[TARGET].std(),
            view[TARGET].quantile(0.9), view[TARGET].quantile(0.99), view[TARGET].max(), view[TARGET].skew(),
        ],
    })
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Rata-rata jauh di atas median dan skewness positif menandakan ekor kanan panjang dari klaim besar.")

with tab_num:
    numeric = [c for c in NUMERIC_NAMES + [DERIVED] if c in view.columns]
    feature = st.selectbox("Fitur", numeric, format_func=lambda c: LABELS.get(c, c))
    log_y = st.toggle("Sumbu Y biaya logaritmik", value=False)
    left, right = st.columns(2)
    with left:
        show(style_fig(px.histogram(
            view, x=feature, nbins=40, color_discrete_sequence=[NEUTRAL],
            labels={feature: LABELS.get(feature, feature)},
        ), 340))
    with right:
        sample = view.sample(min(len(view), 5000), random_state=0)
        show(style_fig(px.scatter(
            sample, x=feature, y=TARGET, opacity=0.35, log_y=log_y, color_discrete_sequence=[ACCENT],
            labels={feature: LABELS.get(feature, feature), TARGET: "Biaya (USD)"},
        ), 340))
    st.caption(
        f"Korelasi Pearson dengan biaya: {fmt_num(view[feature].corr(view[TARGET]), 3)}. "
        "Titik pada grafik adalah sampel acak 5.000 baris."
    )

with tab_cat:
    options = [c for c in CATEGORY_NAMES + BINARY_NAMES if c in view.columns]
    column = st.selectbox("Fitur", options, format_func=lambda c: LABELS.get(c, c))
    groups = view[column].map({0: "Tidak", 1: "Ya"}) if column in BINARY_NAMES else view[column]
    groups = groups.fillna("Tidak tercatat").astype(str)
    table = (
        view.assign(kelompok=groups).groupby("kelompok")[TARGET]
        .agg(n="size", rata_rata="mean", median="median").reset_index()
    )
    long = table.melt("kelompok", ["rata_rata", "median"], var_name="ukuran", value_name="biaya")
    show(style_fig(px.bar(
        long, x="kelompok", y="biaya", color="ukuran", barmode="group",
        color_discrete_map={"rata_rata": ACCENT, "median": NEUTRAL},
        labels={"kelompok": LABELS.get(column, column), "biaya": "Biaya (USD)", "ukuran": ""},
    ), 340))
    st.dataframe(table, hide_index=True, width="stretch")

with tab_corr:
    cols = [c for c in NUMERIC_NAMES + BINARY_NAMES + [DERIVED] if c in view.columns]
    method = st.radio("Metode", ["pearson", "spearman"], horizontal=True)
    with st.spinner("Menghitung korelasi..."):
        corr = view[cols + [TARGET]].corr(method=method)
    with_target = corr[TARGET].drop(TARGET).sort_values()
    show(style_fig(px.bar(
        x=with_target.values, y=[LABELS.get(c, c) for c in with_target.index], orientation="h",
        color_discrete_sequence=[ACCENT], labels={"x": f"Korelasi {method} dengan biaya", "y": ""},
    ), max(320, 22 * len(with_target))))
    top_n = st.slider("Jumlah fitur pada heatmap", 5, min(20, len(with_target)), min(12, len(with_target)))
    top = with_target.abs().sort_values(ascending=False).head(top_n).index.tolist()
    show(style_fig(px.imshow(
        corr.loc[top + [TARGET], top + [TARGET]], text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale=[ALERT, "#FFFFFF", ACCENT],
    ), 520))
    st.caption("Pasangan fitur dengan korelasi tinggi, misalnya rawat inap dan hari rawat, perlu dicek VIF untuk model linier.")

with tab_raw:
    st.dataframe(view.head(500), width="stretch")
    st.caption("Hanya kolom klinis, pemanfaatan layanan, dan target yang dimuat. Identitas dan variabel sosio-ekonomi tidak dibaca.")