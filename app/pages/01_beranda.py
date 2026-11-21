import streamlit as st

from loaders import dataset_path, load_dataset, load_table
from settings import METRICS_PATH, R2_TARGET, TARGET
from ui import demo_banner, fmt_num, fmt_usd, missing_state

st.title("Prediksi Biaya Klaim Medis Tahunan")
st.caption("Berdasarkan kondisi klinis dan pemanfaatan fasilitas kesehatan")
demo_banner()

st.markdown(
    "Asuransi yang salah memperkirakan biaya klaim akan salah menetapkan tarif. Nasabah berisiko "
    "tinggi yang dikenai tarif murah membuat *loss ratio* membengkak (under-pricing), sedangkan "
    "nasabah sehat yang dikenai tarif mahal cenderung pindah ke kompetitor (over-pricing). "
    "Model di sini memprediksi biaya klaim medis tahunan dari data klinis dan riwayat pemanfaatan "
    "layanan. Model tidak memprediksi premi akhir, karena premi bergantung pada faktor komersial."
)

st.header("Model terbaik")
metrics, err = load_table(METRICS_PATH)
needed = {"model", "r2_test", "rmse_test", "mae_test"}
if metrics is None:
    missing_state("Tabel metrik model", METRICS_PATH, err)
elif not needed <= set(metrics.columns):
    st.error(f"Kolom metrik kurang: {', '.join(sorted(needed - set(metrics.columns)))}.")
else:
    best = metrics.loc[metrics["r2_test"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(best["model"]))
    c2.metric("R² data uji", fmt_num(best["r2_test"], 3))
    c3.metric("RMSE data uji", fmt_usd(best["rmse_test"]))
    c4.metric("MAE data uji", fmt_usd(best["mae_test"]))
    if best["r2_test"] > R2_TARGET:
        st.success(f"Target R² di atas {fmt_num(R2_TARGET)} tercapai pada data uji.")
    else:
        st.warning(f"Target R² di atas {fmt_num(R2_TARGET)} belum tercapai pada data uji.")

st.header("Data")
data, err = load_dataset()
if data is None:
    missing_state("Dataset", dataset_path(), err)
elif TARGET not in data.columns:
    st.error(f"Kolom target `{TARGET}` tidak ada di dataset.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah nasabah", fmt_num(len(data), 0))
    c2.metric("Rata-rata biaya", fmt_usd(data[TARGET].mean()))
    c3.metric("Median biaya", fmt_usd(data[TARGET].median()))
    c4.metric("Biaya tertinggi", fmt_usd(data[TARGET].max()))
    st.caption(
        "Sumber: Kaggle, Medical Insurance Cost Prediction (Thalla, 2025), lisensi CC0. "
        "Data Amerika Serikat periode 2015-2024."
    )

st.header("Isi aplikasi")
st.markdown(
    "| Halaman | Untuk | Isi |\n|---|---|---|\n"
    "| Eksplorasi Data | Aktuaria, Data Science | Distribusi biaya, fitur klinis, korelasi |\n"
    "| Prediksi Biaya | Underwriting, calon nasabah | Estimasi satu nasabah atau banyak nasabah dari CSV |\n"
    "| Performa Model | Aktuaria, Manajemen | Perbandingan model, residual, segmen biaya tinggi, feature importance |\n"
    "| Fairness dan Monitoring | Manajemen, Data Science | Error per kelompok dan pemeriksaan drift data baru |"
)

st.header("Batasan penggunaan")
st.markdown(
    "- Hasil adalah estimasi pendukung keputusan, bukan keputusan penerimaan polis atau penetapan premi.\n"
    "- Model dilatih dari data Amerika Serikat 2015-2024. Pada populasi atau periode lain, akurasi dapat turun.\n"
    "- Variabel sosio-ekonomi dan geografis (region, urban_rural, income, education, marital_status) "
    "sengaja tidak dipakai, sesuai batasan etika proyek.\n"
    "- Data kesehatan tergolong data pribadi sensitif. Aplikasi tidak menulis input pengguna ke disk."
)
st.caption(
    "Kelompok: Bylbiss El Haqqie, Muhammad Arif Hermawan, Alifah Chairul Munawar, Ikhsan Wahyu Endriarto. "
    "Dosen pengampu: Dhimas Arief Dharmawan, S.T., Ph.D."
)