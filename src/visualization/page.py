import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

    st.markdown("### 🌐 Visualisasi 2D PCA")
    fig_pca = plot_pca(df, labels)
    st.pyplot(fig_pca)

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

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    visualization_page()
