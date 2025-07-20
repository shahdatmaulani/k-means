import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import os
import glob

# --- Konfigurasi Visual Global ---
def configure_visuals():
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

# --- Load File CSV Terbaru ---
def load_latest_csv(folder_path, pattern="*.csv"):
    csv_files = glob.glob(os.path.join(folder_path, pattern))
    if not csv_files:
        return None, None
    latest_file = max(csv_files, key=os.path.getmtime)
    return pd.read_csv(latest_file), latest_file

# --- Hitung Elbow & Silhouette ---
def calculate_k_metrics(df, max_k):
    distortions, silhouettes = [], []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42).fit(df)
        distortions.append(model.inertia_)
        silhouettes.append(silhouette_score(df, model.labels_))
    return distortions, silhouettes

# --- Plot Elbow & Silhouette ---
def plot_elbow_silhouette(K_range, distortions, silhouettes, chosen_k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, distortions, marker='o', color='#006D77')
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.axvline(chosen_k, color='gray', linestyle='--', alpha=0.5)

    ax2.plot(K_range, silhouettes, marker='o', color='#83C5BE')
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.axvline(chosen_k, color='gray', linestyle='--', alpha=0.5)

    return fig

# --- Visualisasi PCA 2D ---
def plot_pca(df, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='Cluster',
        palette='Set2', s=70, edgecolor='white', alpha=0.9, ax=ax
    )
    ax.set_facecolor('#f9f9f9')
    ax.set_title("PCA Scatter Plot by Cluster")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig

# --- Hitung Mean per Cluster + Count ---
def get_mean_per_cluster(df, labels):
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    mean_df = df_clustered.groupby('Cluster').mean()
    item_count = df_clustered['Cluster'].value_counts().sort_index().rename('Item Count')
    return pd.concat([item_count, mean_df], axis=1)

# --- Profil Numerik dan Kategorikal ---
def get_cluster_profile(df_master, labels, num_cols, cat_cols):
    df = df_master.copy()
    df['Cluster'] = labels

    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None

    profile_num = df.groupby('Cluster')[num_cols].mean()
    profile_cat = df.groupby('Cluster')[cat_cols].agg(get_mode)
    count = df['Cluster'].value_counts().sort_index().rename('Item Count')
    return pd.concat([count, profile_num, profile_cat], axis=1)

# --- Halaman Visualisasi Utama ---
def visualization_page():
    configure_visuals()
    st.title("\U0001F4CA K-Means Clustering Dashboard")

    # Load data encoded dan master
    df, data_path = load_latest_csv("data", "standardized_*.csv")
    df_master_full, master_path = load_latest_csv("data", "dataset_master_*.csv")

    if df is None or df_master_full is None:
        st.error("❌ Data belum lengkap. Pastikan file encoded & master ada di folder 'data/'.")
        return

    st.info(f"📄 Menggunakan file clustering: '{os.path.basename(data_path)}'")
    st.info(f"📄 Menggunakan file master: '{os.path.basename(master_path)}'")

    # Sidebar konfigurasi
    st.sidebar.header("\U0001F527 Clustering Configuration")
    st.sidebar.markdown("### ⚙️ Mode Pemilihan K")
    k_mode = st.sidebar.radio("Pilih metode:", ["Auto", "Manual"])
    max_k = st.sidebar.slider("Max K (Elbow)", 2, 10, 6)
    if k_mode == "Manual":
        chosen_k = st.sidebar.slider("Chosen K (Clustering)", 2, max_k, 4)
    else:
        chosen_k = None

    # Filter Target Audience
    target_options = ["All"] + sorted(df_master_full['Target Audience'].dropna().unique())
    selected_audience = st.sidebar.selectbox("\U0001F3AF Filter Target Audience", target_options)

    if selected_audience != "All":
        mask = df_master_full['Target Audience'] == selected_audience
        df_master_filtered = df_master_full[mask].reset_index(drop=True)
        df = df.loc[mask].reset_index(drop=True)
        df_master_full = df_master_filtered
        st.success(f"Menampilkan hasil visualisasi hanya untuk Target Audience: **{selected_audience}**")
    else:
        st.info("Menampilkan hasil visualisasi untuk seluruh data.")

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

    st.markdown("### \U0001F310 Visualisasi 2D PCA")
    fig_pca = plot_pca(df, labels)
    st.pyplot(fig_pca)

    # Mean per Cluster
    st.markdown("### \U0001F4CA Rata-rata Fitur per Cluster")
    mean_df = get_mean_per_cluster(df, labels)
    st.dataframe(mean_df, use_container_width=True)

    # Profil Lengkap dari Master
    st.markdown("### \U0001F9FE Data Dominan per Cluster")
    num_cols = ['Price ($)', 'Width (Inch)', 'Height (inch)']
    cat_cols = ['Style', 'Color Palette', 'Mood/Atmosphere', 'Theme/Lighting Requirements', 'Target Audience']
    profile_full = get_cluster_profile(df_master_full, labels, num_cols, cat_cols)
    st.dataframe(profile_full, use_container_width=True)

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    visualization_page()
