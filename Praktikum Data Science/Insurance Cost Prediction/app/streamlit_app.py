import streamlit as st

st.set_page_config(page_title="Prediksi Biaya Klaim Medis", layout="wide")

pages = [
    st.Page("pages/1_Beranda.py", title="Beranda", default=True),
    st.Page("pages/2_Eksplorasi_Data.py", title="Eksplorasi Data"),
    st.Page("pages/3_Prediksi_Biaya.py", title="Prediksi Biaya"),
    st.Page("pages/4_Performa_Model.py", title="Performa Model"),
    st.Page("pages/5_Fairness_Monitoring.py", title="Fairness dan Monitoring"),
]

st.sidebar.caption("Data Science IF-B, UPN Veteran Yogyakarta")
st.navigation(pages).run()