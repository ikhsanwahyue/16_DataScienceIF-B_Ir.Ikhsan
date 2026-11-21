# Prediksi Biaya Klaim Medis Tahunan Berdasarkan Kondisi Klinis dan Pemanfaatan Fasilitas Kesehatan

Proyek akhir mata kuliah **Data Science** (kelas Teori dan Praktikum), Program Studi Informatika, Fakultas Teknik Industri, Universitas Pembangunan Nasional "Veteran" Yogyakarta, Tahun Ajaran Ganjil 2026/2027.

Proyek ini membangun model **regresi** untuk memprediksi biaya klaim medis tahunan (`annual_medical_cost`) seorang nasabah asuransi kesehatan dari kondisi klinis dan riwayat pemanfaatan fasilitas kesehatannya. Seluruh alur kerja mengikuti kerangka CRISP-DM dan dikerjakan dalam satu notebook (`main.ipynb`), dilengkapi dashboard Streamlit untuk mencoba model.

## Daftar Isi

1. [Ringkasan Proyek](#ringkasan-proyek)
2. [Anggota dan Kontribusi](#anggota-dan-kontribusi)
3. [Sumber Data](#sumber-data)
4. [Alur Pengerjaan](#alur-pengerjaan)
5. [Struktur Repository](#struktur-repository)
6. [Cara Menjalankan](#cara-menjalankan)
7. [Dashboard Streamlit](#dashboard-streamlit)
8. [Asumsi, Batasan, dan Etika](#asumsi-batasan-dan-etika)
9. [Lisensi dan Referensi](#lisensi-dan-referensi)

## Ringkasan Proyek

**Masalah.** Perusahaan asuransi kesehatan yang tidak dapat memperkirakan biaya klaim nasabah dengan akurat akan salah menetapkan premi (*mispricing*):

- *Under-pricing*: nasabah berisiko tinggi dikenai tarif murah, sehingga klaim melampaui premi dan *loss ratio* membengkak.
- *Over-pricing*: nasabah sehat dikenai tarif mahal, sehingga cenderung berpindah ke kompetitor (*customer churn*).

**Solusi.** Model regresi yang mengestimasi **biaya tagihan medis tahunan** secara objektif dari data klinis dan pemanfaatan layanan. Biaya tagihan murni dipilih sebagai target, bukan premi akhir, karena premi dipengaruhi faktor komersial seperti tier polis dan margin perusahaan.

**Pemangku kepentingan.** Tim aktuaria, tim underwriting, tim pemasaran dan pengembangan produk, tim data science dan IT, manajemen eksekutif, serta calon nasabah.

**Metode dan metrik.**

| Aspek | Keterangan |
|---|---|
| Jenis tugas | Supervised learning, regresi |
| Target | `annual_medical_cost` (USD) |
| Model yang dibandingkan | Linear Regression, Ridge, Lasso |
| Metrik evaluasi | MAE, RMSE, R² |
| Pembagian data | 80% latih dan 20% uji, `random_state=42` |

Hasil, tabel perbandingan, dan kesimpulan lengkap ada pada bagian *Modeling*, *Evaluation*, dan konklusi di `main.ipynb`.

## Anggota dan Kontribusi

**Kelompok** (Informatika, UPN "Veteran" Yogyakarta)

| No | Nama | NIM |
|---|---|---|
| 1 | Bylbiss El Haqqie | 123240003 |
| 2 | Muhammad Arif Hermawan | 123240063 |
| 3 | Alifah Chairul Munawar | 123240234 |
| 4 | Ikhsan Wahyu Endriarto | 123240241 |

**Dosen pengampu:** Dhimas Arief Dharmawan, S.T., Ph.D.

**Pembagian pengerjaan pada `main.ipynb`**

| Tahap | Bagian | Penanggung jawab |
|---|---|---|
| Business Understanding | Latar belakang, solusi, import library, memuat dataset | Muhammad Arif Hermawan |
| Data Understanding | Cek missing value, cek data duplikat, deteksi outlier | Muhammad Arif Hermawan |
| Data Understanding | Exploratory Data Analysis | Bylbiss El Haqqie |
| Data Preparation | Data cleaning | Alifah Chairul Munawar |
| Data Preparation | Penanganan usia minimum | Bylbiss El Haqqie |
| Data Preparation | Penanganan inkonsistensi Age vs Education | Alifah Chairul Munawar |
| Data Preparation | Penanganan inkonsistensi Age vs Marital Status | Muhammad Arif Hermawan |
| Data Preparation | Penanganan inkonsistensi Age vs Employment Status | Ikhsan Wahyu Endriarto |
| Data Preparation | Feature selection dan data splitting | Ikhsan Wahyu Endriarto |
| Data Preparation | Data preprocessing | Ikhsan Wahyu Endriarto |
| Modeling | Pelatihan model, perbandingan model | Bylbiss El Haqqie |
| Evaluation | Metrik evaluasi, evaluasi pada data uji, visualisasi aktual vs prediksi, analisis residual, pemilihan model terbaik | Alifah Chairul Munawar |
| Deployment (di luar lingkup praktikum) | Menyimpan model, membuat dashboard Streamlit | Ikhsan Wahyu Endriarto |

## Sumber Data

| Item | Keterangan |
|---|---|
| Nama dataset | Medical Insurance Cost Prediction |
| Penyedia | Mohan Krishna Thalla, melalui Kaggle |
| Tautan | https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction |
| Lisensi | CC0: Public Domain |
| Cakupan | Data individu di Amerika Serikat periode 2015-2024, menurut deskripsi pada halaman dataset |
| Ukuran asli | 100.000 baris dan 54 kolom |
| File di repository | `data/medical_insurance.csv` |

Dataset memuat profil demografi, status sosial-ekonomi, kondisi kesehatan, gaya hidup, informasi asuransi, dan pemanfaatan fasilitas kesehatan.

**Ringkasan kondisi data.**

- Missing value hanya pada `alcohol_freq` (sekitar 30%), ditangani dengan kategori `Unknown`.
- Tidak ditemukan baris duplikat.
- Banyak kolom numerik berdistribusi miring ke kanan dan memiliki outlier. Target `annual_medical_cost` sangat miring ke kanan (skewness sekitar 4).
- Ditemukan inkonsistensi antar kolom (usia terhadap pendidikan, status pernikahan, dan status pekerjaan) yang ditangani pada tahap Data Preparation.

**Pemilihan fitur.** Model memakai fitur klinis dan pemanfaatan fasilitas kesehatan. Kolom identitas, variabel sosio-ekonomi dan geografis, atribut polis dan komersial, serta variabel turunan yang berisiko menyebabkan kebocoran data (misalnya premi, jumlah dan nilai klaim, `risk_score`) dikeluarkan. Alasan setiap keputusan dijelaskan pada bagian *Feature Selection* di notebook.

## Alur Pengerjaan

Seluruh proses ada di satu notebook, `main.ipynb`, dengan urutan:

1. **Business Understanding**: masalah, solusi, tujuan, dan metrik.
2. **Data Understanding**: kualitas data (missing value, duplikat, outlier) dan EDA.
3. **Data Preparation**: pembersihan, penanganan inkonsistensi, pemilihan fitur, pembagian data, encoding, dan scaling.
4. **Modeling**: pelatihan dan perbandingan model regresi.
5. **Evaluation**: metrik, grafik aktual vs prediksi, analisis residual, dan pemilihan model terbaik.
6. **Deployment**: penyimpanan model beserta hasil evaluasinya ke `models/model.joblib` untuk dashboard.

## Struktur Repository

```
InsuranceCostPrediction-DataScience/
├── .streamlit/
│   └── config.toml                  tema dashboard
├── app/
│   ├── streamlit_app.py             titik masuk dan navigasi
│   ├── settings.py                  path dan konstanta
│   ├── schema.py                    skema form input
│   ├── loaders.py                   pemuat model dan data
│   ├── modeling.py                  fungsi prediksi dan analisis
│   ├── ui.py                        komponen tampilan
│   └── pages/
│       ├── 01_beranda.py
│       ├── 02_ekplorasi_data.py
│       ├── 03_prediksi_biaya.py
│       ├── 04_peforma_model.py
│       └── 05_fairness_monitoring.py
├── data/
│   └── medical_insurance.csv        dataset
├── models/
│   └── model.joblib                 bundle model dan hasil evaluasi (dibuat main.ipynb)
├── config.yaml
├── LICENSE
├── main.ipynb                       notebook utama (BU sampai konklusi)
├── README.md
└── requirements.txt
```

## Cara Menjalankan

### Prasyarat

- Python 3.10 atau lebih baru.
- Git.
- Lingkungan yang dapat menjalankan Jupyter Notebook (VS Code atau Jupyter).

### 1. Unduh proyek dan pasang library

```bash
git clone https://github.com/ikhsanwahyue/16_DataScienceIF-B_Ir.Ikhsan.git
cd 16_DataScienceIF-B_Ir.Ikhsan

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Jalankan notebook

1. Buka `main.ipynb` dan pilih kernel dari lingkungan `.venv`.
2. Pilih **Restart** lalu **Run All**.
3. Sel terakhir menyimpan `models/model.joblib`. Pastikan tercetak pesan "Model tersimpan".

Dataset dibaca dari `data/medical_insurance.csv`. Hasil dapat direproduksi karena pembagian data memakai `random_state=42`.

### 3. Jalankan dashboard

Dari folder root repository:

```bash
streamlit run app/streamlit_app.py
```

Dashboard terbuka di `http://localhost:8501`. Jalankan dari folder root agar path data, model, dan tema terbaca. Jika `models/model.joblib` belum ada, halaman terkait menampilkan pesan "belum tersedia", bukan error.

> Versi scikit-learn pada lingkungan dashboard harus sama dengan yang dipakai saat menjalankan notebook, karena file `.joblib` bergantung pada versi library.

## Dashboard Streamlit

| Halaman | Isi |
|---|---|
| Beranda | Ringkasan proyek, metrik model terbaik, statistik data, batasan penggunaan |
| Eksplorasi Data | Distribusi biaya, fitur numerik dan kategorikal, korelasi, dengan filter interaktif |
| Prediksi Biaya | Estimasi biaya untuk satu nasabah lewat form, atau banyak nasabah lewat unggah CSV |
| Performa Model | Perbandingan model, aktual vs prediksi, residual, segmen biaya tinggi, feature importance |
| Fairness dan Monitoring | Error model per jenis kelamin dan kelompok usia, serta pemeriksaan pergeseran data (drift) pada data baru |

Semua metrik pada dashboard dibaca dari `models/model.joblib`, yaitu hasil langsung dari `main.ipynb`. Dashboard bukan bagian yang dinilai pada kelas Praktikum dan hanya berfungsi sebagai rencana penerapan untuk kelas Teori.

## Asumsi, Batasan, dan Etika

- Model dilatih dari data Amerika Serikat periode 2015-2024. Penerapan pada populasi atau periode lain dapat menurunkan akurasi, dan model perlu dilatih ulang secara berkala.
- Hubungan antara kondisi klinis dan biaya diasumsikan relatif stabil selama periode data.
- Variabel sosio-ekonomi dan geografis sengaja tidak dipakai agar model tidak mempelajari diskriminasi historis.
- Data kesehatan tergolong data pribadi yang sensitif (45 CFR § 164.502(b)). Kolom identitas tidak dipakai, dan dashboard tidak menyimpan data yang diunggah pengguna.
- Hasil model adalah estimasi pendukung keputusan, bukan keputusan akhir penerimaan polis atau penetapan premi. 

## Lisensi dan Referensi

- Kode proyek: lihat berkas [LICENSE](LICENSE).
- Dataset: CC0 (Public Domain).

**Referensi**

Thalla, M. K. (2025). *Medical Insurance Cost Prediction*. Kaggle. https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction

TODO:
- Tambah kamus data di README.md
- Streamlit: model memprediksi terlalu tinggi