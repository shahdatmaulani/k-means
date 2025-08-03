import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from src.visualization.services import (
    auto_select_k, run_clustering, calculate_k_metrics
)
from src.visualization.clustering import get_mean_per_cluster
from src.visualization.plotting import plot_elbow_silhouette
from src.config import configure_visuals, DATA_DIR
from src.standardization.io_utils import load_latest_csv


def visualization_page():
    configure_visuals()
    st.title("📊 K-Means Clustering Dashboard")

    # Load data
    df, _ = load_latest_csv(DATA_DIR, "standardized_*.csv")
    df_master_full, _ = load_latest_csv(DATA_DIR, "dataset_master_*.csv")

    if df is None or df_master_full is None:
        st.error("❌ Data belum lengkap. Pastikan file encoded & master ada di folder 'data/'.")
        return

    # Sidebar konfigurasi
    st.sidebar.header("🛠️ Clustering Configuration")
    k_mode = st.sidebar.radio("Pilih metode pemilihan K:", ["Auto", "Manual"])
    max_k = st.sidebar.slider("Max K (Elbow)", 2, 10, 10)
    chosen_k = None if k_mode == "Auto" else st.sidebar.slider("Chosen K", 2, max_k, 4)

    # Filter Target Audience
    selected_audience = None
    if "Target Audience" in df_master_full.columns:
        target_values = df_master_full['Target Audience'].dropna().unique()
        if len(target_values) > 0:
            target_options = ["All"] + sorted(target_values)
            choice = st.sidebar.selectbox("🎯 Filter Target Audience", target_options)
            if choice != "All":
                selected_audience = choice
                st.success(f"Menampilkan hasil untuk Target Audience: **{choice}**")
            else:
                st.info("🌐 Menampilkan hasil untuk seluruh data.")
        else:
            st.info("ℹ️ Kolom 'Target Audience' kosong. Lanjut clustering seluruh data.")
    else:
        st.info("ℹ️ Kolom 'Target Audience' tidak ditemukan.")

    # Hitung metrics
    K_range = range(2, max_k + 1)
    distortions, silhouettes = calculate_k_metrics(df, max_k)

    if k_mode == "Auto":
        chosen_k, silhouettes = auto_select_k(df, max_k=max_k)
        st.success(f"✅ Auto memilih K = {chosen_k}")
    else:
        st.info(f"📌 Menggunakan K = {chosen_k}")

    # Plot Elbow & Silhouette
    fig_elbow = plot_elbow_silhouette(K_range, distortions, silhouettes, chosen_k)
    st.markdown("### 📉 Elbow Method & 📈 Silhouette Score")
    st.pyplot(fig_elbow)

    # Jalankan clustering
    try:
        result = run_clustering(df, df_master_full, chosen_k, audience=selected_audience)
        labels = result["labels"]
        df_filtered = result["df_filtered"]   # ✅ dataframe hasil filter
    except ValueError as e:
        st.error(str(e))
        return

    # Hasil clustering
    df_clustered = df_filtered.copy()
    df_clustered["Cluster"] = labels

    cols = ["Cluster"] + [c for c in df_clustered.columns if c != "Cluster"]
    df_clustered = df_clustered[cols]

    if st.checkbox(f"📋 Lihat seluruh data hasil clustering (K = {chosen_k})"):
        st.dataframe(df_clustered, use_container_width=True)

    # Mean per cluster
    st.markdown("### 📊 Rata-rata Fitur per Cluster")
    mean_df = get_mean_per_cluster(df_filtered, labels)   # ✅ pakai df_filtered
    st.dataframe(mean_df, use_container_width=True)

    # Profil cluster
    st.markdown("### 🧾 Data Dominan per Cluster")
    num_cols = ['Price ($)', 'Width (Inch)', 'Height (inch)']
    cat_cols = ['Style', 'Color Palette', 'Mood/Atmosphere',
                'Theme/Lighting Requirements', 'Target Audience']
    st.dataframe(result["profile"], use_container_width=True)

    # PCA Summary
    n_components = df_filtered.shape[1]   # ✅ pakai df_filtered
    pca_model = PCA(n_components=n_components).fit(df_filtered)

    pca_summary = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Standard Deviation": np.sqrt(pca_model.explained_variance_),
        "Proportion of Variance": pca_model.explained_variance_ratio_,
        "Cumulative Variance": pca_model.explained_variance_ratio_.cumsum()
    })

    if st.checkbox("🔎 Tampilkan ringkasan PCA"):
        st.dataframe(pca_summary, use_container_width=True)

    # PCA Custom Scatter
    st.markdown("### 🎨 Custom PCA Visualization")
        
    components = pca_model.transform(df_filtered)   # ✅ pakai df_filtered
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["Cluster"] = labels
    pca_df["ID"] = range(1, len(df_filtered)+1)   # ✅ panjang sesuai df_filtered

    axis_options = list(pca_df.columns)
    x_axis = st.selectbox("Sumbu X", axis_options, index=axis_options.index("ID"))
    y_axis = st.selectbox("Sumbu Y", axis_options, index=axis_options.index("PC2"))
    color_axis = st.selectbox("Pewarnaan", axis_options, index=axis_options.index("Cluster"))

    fig_custom = px.scatter(
        pca_df, x=x_axis, y=y_axis, color=color_axis,
        hover_data=["ID", "Cluster"],
        template="plotly_white",
        title=f"Custom PCA: {x_axis} vs {y_axis} (colored by {color_axis})"
    )
    st.plotly_chart(fig_custom, use_container_width=True)