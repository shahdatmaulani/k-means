import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import os
import glob

def visualization_page():
    st.title("📊 K-Means Clustering Dashboard")

    # Cari file .csv terbaru (hasil standardisasi)
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if not csv_files:
        st.error("❌ Tidak ditemukan file .csv di folder 'data/'.")
        return

    # Gunakan file terbaru (asumsi data yang distandarisasi)
    data_path = max(csv_files, key=os.path.getmtime)
    st.info(f"📄 Menggunakan file clustering '{os.path.basename(data_path)}'")
    df = pd.read_csv(data_path)

    # Sidebar: pilih K
    st.sidebar.header("🔧 Clustering Configuration")
    max_k = st.sidebar.slider("Max K (Elbow)", 2, 10, 6)
    chosen_k = st.sidebar.slider("Chosen K (Clustering)", 2, max_k, 4)

    # 1. Elbow Chart
    st.markdown("### 📈 Elbow Chart")
    distortions = []
    K = range(2, max_k + 1)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42).fit(df)
        distortions.append(km.inertia_)
    fig1, ax1 = plt.subplots()
    ax1.plot(K, distortions, marker='o')
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method for Optimal K")
    st.pyplot(fig1)

    # 2. Clustering dan hasil
    kmeans = KMeans(n_clusters=chosen_k, random_state=42).fit(df)
    labels = kmeans.labels_
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    # 3. PCA Plot
    st.markdown("### 🌀 PCA 2D Scatter Plot")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax2)
    ax2.set_title("PCA Scatter Plot by Cluster")
    st.pyplot(fig2)

    # 4. Mean per Cluster
    st.markdown("### 📊 Mean of Features per Cluster")
    mean_df = df_clustered.groupby('Cluster').mean()
    st.dataframe(mean_df)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    mean_df.iloc[:, :5].T.plot(kind='bar', ax=ax3)
    ax3.set_title("Mean Values of First 5 Features by Cluster")
    st.pyplot(fig3)

    # 5. Summary Table
    st.markdown("### 📋 Cluster Summary Table")
    summary = df_clustered.groupby('Cluster').agg(['count', 'mean']).T
    st.dataframe(summary)

    # 6. Silhouette Score
    st.markdown("### 📏 Silhouette Score")
    score = silhouette_score(df, labels)
    st.success(f"Silhouette Score for K = {chosen_k}: **{score:.3f}**")

    # 7. Profil Lengkap (gabung ke dataset_master terbaru)
    st.markdown("### 🧾 Profil Lengkap per Cluster")

    # Cari dataset_master terbaru
    master_files = glob.glob(os.path.join("data", "dataset_master_*.csv"))
    if not master_files:
        st.warning("⚠️ File `dataset_master_*.csv` tidak ditemukan.")
        return

    master_path = max(master_files, key=os.path.getmtime)
    # st.info(f"📄 Menggunakan file master '{os.path.basename(master_path)}'")
    df_master = pd.read_csv(master_path)

    # Gabungkan cluster label ke dataset_master
    df_master_with_cluster = df_master.copy()
    df_master_with_cluster['Cluster'] = labels

    # Definisikan kolom
    num_cols = ['Price ($)', 'Width (Inch)', 'Height (inch)']
    cat_cols = ['Style', 'Color Palette', 'Mood/Atmosphere', 'Theme/Lighting Requirements', 'Target Audience']

    # Hitung profil
    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None

    profile_num = df_master_with_cluster.groupby('Cluster')[num_cols].mean()
    profile_cat = df_master_with_cluster.groupby('Cluster')[cat_cols].agg(get_mode)
    count_per_cluster = df_master_with_cluster['Cluster'].value_counts().sort_index().rename('Item Count')

    profile_full = pd.concat([count_per_cluster, profile_num, profile_cat], axis=1)
    st.dataframe(profile_full)
