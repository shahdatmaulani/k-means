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

    # Cari file CSV terbaru di folder 'data/'
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if not csv_files:
        st.error("❌ Tidak ditemukan file .csv di folder 'data/'.")
        return

    data_path = max(csv_files, key=os.path.getmtime)
    st.info(f"📄 Menggunakan file terbaru: `{os.path.basename(data_path)}`")

    # Load dataset
    df = pd.read_csv(data_path)

    # Sidebar: user set K
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

    # Fit KMeans with chosen_k
    kmeans = KMeans(n_clusters=chosen_k, random_state=42).fit(df)
    labels = kmeans.labels_
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    # 2. PCA Scatter Plot
    st.markdown("### 🌀 PCA 2D Scatter Plot")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax2)
    ax2.set_title("PCA Scatter Plot by Cluster")
    st.pyplot(fig2)

    # 3. Bar Chart: Mean per Cluster
    st.markdown("### 📊 Mean of Features per Cluster")
    mean_df = df_clustered.groupby('Cluster').mean()
    st.dataframe(mean_df)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    mean_df.iloc[:, :5].T.plot(kind='bar', ax=ax3)
    ax3.set_title("Mean Values of First 5 Features by Cluster")
    ax3.set_ylabel("Standardized Mean")
    st.pyplot(fig3)

    # 4. Cluster Summary Table
    st.markdown("### 📋 Cluster Summary Table")
    summary = df_clustered.groupby('Cluster').agg(['count', 'mean']).T
    st.dataframe(summary)

    # 5. Silhouette Score
    st.markdown("### 📏 Silhouette Score")
    score = silhouette_score(df, labels)
    st.success(f"Silhouette Score for K = {chosen_k}: **{score:.3f}**")
