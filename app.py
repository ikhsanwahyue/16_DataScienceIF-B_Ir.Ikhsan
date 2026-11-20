import nbformat
from nbconvert import HTMLExporter
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Prediksi Biaya Asuransi",
    layout="wide",
)

mode = st.sidebar.selectbox("Select Mode", ["Notebook", "Prediksi Biaya Asuransi"])

if mode == "Notebook":
    st.title("Notebook Viewer")
    with open("main.ipynb", "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.template_name = "lab"
    body, _ = html_exporter.from_notebook_node(nb)

    components.html(body, height=800, scrolling=True)

if mode == "Prediksi Biaya Asuransi":
    # TODO: buat halaman untuk prediksi biaya asuransi, misal input 29 kolom (berdasrkan feature selection), kemudian ada tombol prediksi, dan tampilkan hasil prediksi biaya asuransi. Gunakan joblib untuk memanggil model Lasso yang sudah disimpan, jangan latih model baru disini.
    print()