import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

from src.visualization.clustering import (
    calculate_k_metrics, get_mean_per_cluster, get_cluster_profile
)
from src.visualization.plotting import (
    plot_elbow_silhouette, plot_pca
)
from src.config import configure_visuals
from src.standardization.io_utils import load_latest_csv

def visualization_page():
    configure_visuals()
    st.title("📊 K-Means Clustering Dashboard")

    # Load data encoded dan master
    df, data_path = load_latest_csv("data", "standardized_*.csv")
    df_master_full, master_path = load_latest_csv("data", "dataset_master_*.csv")

    if df is None or df_master_full is None:
        st.error("❌ Data belum lengkap. Pastikan file encoded & master ada di folder 'data/'.")
        return
    
    #st.markdown(f"<small>📄 Menggunakan file clustering: '{os.path.basename(data_path)}'</small>", unsafe_allow_html=True)
    #st.markdown(f"<small>📄 Menggunakan file master: '{os.path.basename(master_path)}'</small>", unsafe_allow_html=True)

    # Sidebar konfigurasi
    st.sidebar.header("🛠️ Clustering Configuration")
    st.sidebar.markdown("### ⚙️ Mode Pemilihan K")
    k_mode = st.sidebar.radio("Pilih metode:", ["Auto", "Manual"])
    max_k = st.sidebar.slider("Max K (Elbow)", 2, 10, 6)
    if k_mode == "Manual":
        chosen_k = st.sidebar.slider("Chosen K (Clustering)", 2, max_k, 4)
    else:
        chosen_k = None

    # Filter Target Audience
    if "Target Audience" in df_master_full.columns:
        target_options = ["All"] + sorted(df_master_full['Target Audience'].dropna().unique())
        selected_audience = st.sidebar.selectbox("🎯 Filter Target Audience", target_options)

        if selected_audience != "All":
            mask = df_master_full['Target Audience'] == selected_audience
            df_master_filtered = df_master_full[mask].reset_index(drop=True)
            df = df.loc[mask].reset_index(drop=True)
            df_master_full = df_master_filtered
            st.success(f"Menampilkan hasil visualisasi Target Audience: **{selected_audience}**")
        else:
            st.info("🌐 Menampilkan hasil visualisasi untuk seluruh data.")
    else:
        st.info("ℹ️ Kolom 'Target Audience' tidak ditemukan, visualisasi ditampilkan untuk seluruh data.")

    # Hitung Elbow dan Silhouette
    K_range = range(2, max_k + 1)
    distortions, silhouettes = calculate_k_metrics(df, max_k)

    if k_mode == "Auto":
        chosen_k = K_range[np.argmax(silhouettes)]
        st.success(f"✅ Otomatis memilih K = {chosen_k} (Silhouette Score: `{silhouettes[chosen_k - 2]:.3f}`)")
    else:
        st.info(f"📌 Menggunakan K = {chosen_k} (Silhouette Score: `{silhouettes[chosen_k - 2]:.3f}`)")

    fig_elbow = plot_elbow_silhouette(K_range, distortions, silhouettes, chosen_k)
    st.markdown("### 📉 Elbow Method & 📈 Silhouette Score")
    st.pyplot(fig_elbow)
    st.markdown(f"📉 **Inertia untuk K = {chosen_k}: `{distortions[chosen_k - 2]:,.0f}`**")

    with st.expander("📋 Lihat semua nilai Silhouette per K"):
        df_k_summary = pd.DataFrame({
            "K": list(K_range),
            "Inertia": distortions,
            "Silhouette Score": silhouettes
        })
        st.dataframe(df_k_summary, use_container_width=True)

    # Clustering dan PCA
    model = KMeans(n_clusters=chosen_k, random_state=42).fit(df)
    labels = model.labels_

    # Tambahkan kolom cluster ke dataframe standar (supaya bisa ditampilkan)
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    # Tampilkan tabel hasil clustering
    if st.checkbox(f"📋 Tampilkan seluruh data hasil clustering dengan K = {chosen_k}"):
        st.dataframe(df_clustered, use_container_width=True)

    # Mean per Cluster
    st.markdown("### 📊 Rata-rata Fitur per Cluster")
    mean_df = get_mean_per_cluster(df, labels)
    st.dataframe(mean_df, use_container_width=True)

    # Profil Lengkap dari Master
    st.markdown("### 🧾 Data Dominan per Cluster")
    num_cols = ['Price ($)', 'Width (Inch)', 'Height (inch)']
    cat_cols = ['Style', 'Color Palette', 'Mood/Atmosphere', 'Theme/Lighting Requirements', 'Target Audience']
    profile_full = get_cluster_profile(df_master_full, labels, num_cols, cat_cols)
    st.dataframe(profile_full, use_container_width=True)

    # --- PCA Summary (komponen, std dev, variance explained) ---
    # --- PCA Summary (komponen, std dev, variance explained) ---
    n_components = df.shape[1]   # pakai semua kolom dataset
    pca_model = PCA(n_components=n_components)
    pca_model.fit(df)

    pca_summary = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Standard Deviation": np.sqrt(pca_model.explained_variance_),
        "Proportion of Variance": pca_model.explained_variance_ratio_,
        "Cumulative Variance": pca_model.explained_variance_ratio_.cumsum()
    })

    if st.checkbox(f"🔎 Tampilkan ringkasan PCA (PC1 - PC{n_components})"):
        st.dataframe(pca_summary, use_container_width=True)


    # --- Visualisasi PCA Custom ---
    st.markdown("### 🎨 Custom PCA Visualization")
    components = pca_model.transform(df)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["Cluster"] = labels
    pca_df["ID"] = range(1, len(df)+1)

    axis_options = list(pca_df.columns)
    # Cari posisi default untuk ID, PC2, Cluster
    default_x = axis_options.index("ID") if "ID" in axis_options else 0
    default_y = axis_options.index("PC2") if "PC2" in axis_options else 1
    default_color = axis_options.index("Cluster") if "Cluster" in axis_options else 2

    x_axis = st.selectbox("Pilih Sumbu X", axis_options, index=default_x)
    y_axis = st.selectbox("Pilih Sumbu Y", axis_options, index=default_y)
    color_axis = st.selectbox("Pilih Pewarnaan (Color)", axis_options, index=default_color)

    fig_custom = px.scatter(
        pca_df, x=x_axis, y=y_axis, color=color_axis,
        hover_data=["ID", "Cluster"],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        template="plotly_white",
        title=f"Custom PCA Visualization: {x_axis} vs {y_axis} (colored by {color_axis})"
    )
    st.plotly_chart(fig_custom, use_container_width=True)

    # PCA Visualisasi
    #st.markdown("### 🌐 Visualisasi 2D PCA")
    #fig_pca = plot_pca(df, labels)
    #st.pyplot(fig_pca)

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    visualization_page()
