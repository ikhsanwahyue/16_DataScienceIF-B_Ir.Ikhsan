import streamlit as st

st.set_page_config(page_title="Prediksi Biaya Klaim Medis", layout="wide")

pages = [
    st.Page("pages/01_beranda.py", title="Beranda", default=True),
    st.Page("pages/02_ekplorasi_data.py", title="Eksplorasi Data"),
    st.Page("pages/03_prediksi_biaya.py", title="Prediksi Biaya"),
    st.Page("pages/04_peforma_model.py", title="Performa Model"),
    st.Page("pages/05_fairness_monitoring.py", title="Fairness dan Monitoring"),
]

st.sidebar.caption("Data Science IF-B, UPN Veteran Yogyakarta")
st.navigation(pages).run()
