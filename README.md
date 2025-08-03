# 📊 K-Means Dashboard

Proyek ini menyediakan **dashboard interaktif (Streamlit)** dan **CLI (Command Line Interface)** untuk melakukan:
- Standardisasi dataset (rename kolom, one-hot encoding, scaling numerik)
- Clustering dengan algoritma **K-Means**
- Visualisasi hasil clustering (Elbow, Silhouette, PCA)

## 🚀 Cara Menjalankan
### Streamlit
```bash
streamlit run app.py
```
### CLI
```bash
python cli.py
```

## 🔑 Fitur Utama
- Standardisasi data otomatis (rename, encode, scale)
- Clustering K-Means (manual/auto Silhouette)
- Visualisasi Elbow, Silhouette, PCA
- Profil cluster numerik & kategorikal

## 🛠️ Requirement
```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
openpyxl
xlsxwriter
```
