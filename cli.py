import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.standardization.transform import (
    rename_common_columns, encode_categorical_columns, standardize_selected_columns
)
from src.visualization.clustering import get_cluster_profile
from src.standardization.io_utils import save_dataframe_with_timestamp
from src.config import PREFERRED_COLUMNS


# --- Fungsi Standardization ---
def run_standardization(file_name):
    if not os.path.exists(file_name):
        print(f"❌ File {file_name} tidak ditemukan.")
        return

    print(f"🔹 Membaca file: {file_name}")
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_name)
    else:
        print("❌ Format file tidak dikenali (hanya mendukung CSV/XLSX).")
        return

    # Rename kolom
    rename_common_columns(df)

    # One-hot encoding
    df_encoded, _, _ = encode_categorical_columns(df)

    # --- Simpan hasil encoded (tanpa standardisasi) ---
    os.makedirs("data", exist_ok=True)
    encoded_file, _ = save_dataframe_with_timestamp(df_encoded, "encoded_with_numeric")
    master_file, _ = save_dataframe_with_timestamp(df, "dataset_master")
    print(f"✅ File encoded tersimpan: {encoded_file}")
    print(f"✅ File master tersimpan: {master_file}")

    # Standardisasi kolom numerik
    numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cols_to_standardize = [col for col in PREFERRED_COLUMNS if col in numeric_cols]

    if cols_to_standardize:
        df_encoded = standardize_selected_columns(df_encoded, cols_to_standardize)
        print(f"✅ Distandarisasi kolom: {cols_to_standardize}")
    else:
        print("⚠️ Tidak ada kolom numerik untuk standardisasi. Hanya hasil encoding yang digunakan.")

    # --- Simpan hasil standardisasi ---
    standardized_file, _ = save_dataframe_with_timestamp(df_encoded, "standardized_data")
    print(f"✅ File standardized tersimpan: {standardized_file}")


# --- Cari K terbaik dengan Silhouette Score ---
def auto_select_k(df, max_k=10):
    silhouettes = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42).fit(df)
        score = silhouette_score(df, model.labels_)
        silhouettes.append(score)
    best_k = K_range[np.argmax(silhouettes)]
    print(f"✅ Auto memilih K = {best_k} (Silhouette Score = {max(silhouettes):.3f})")
    return best_k


# --- Fungsi Visualization (Data Dominan per Cluster) ---
def run_clustering(k, audience=None):
    data_files = sorted([f for f in os.listdir("data") if f.startswith("standardized_data")], reverse=True)
    master_files = sorted([f for f in os.listdir("data") if f.startswith("dataset_master")], reverse=True)

    if not data_files or not master_files:
        print("❌ Data standarized/master tidak ditemukan. Jalankan standardization dulu.")
        return

    df = pd.read_csv(os.path.join("data", data_files[0]))
    df_master = pd.read_csv(os.path.join("data", master_files[0]))

    # Filter target audience jika ada
    if "Target Audience" in df_master.columns and audience:
        mask = df_master["Target Audience"] == audience
        df = df.loc[mask].reset_index(drop=True)
        df_master = df_master.loc[mask].reset_index(drop=True)
        print(f"🎯 Difilter untuk Target Audience: {audience}")
    else:
        print("🌐 Menggunakan seluruh data.")

    # Jalankan clustering
    model = KMeans(n_clusters=k, random_state=42).fit(df)
    labels = model.labels_

    # Profil cluster
    num_cols = [c for c in df_master.select_dtypes(include=['int64', 'float64']).columns]
    cat_cols = [c for c in df_master.select_dtypes(include=['object', 'category']).columns]

    profile = get_cluster_profile(df_master, labels, num_cols, cat_cols)
    print("📊 Data Dominan per Cluster:")
    print(profile)


# --- Menu Utama ---
def main():
    while True:
        print("\n=== MENU UTAMA ===")
        print("1. Standardization")
        print("2. Visualization (Data Dominan per Cluster)")
        print("3. Keluar")
        choice = input("Pilih [1-3]: ").strip()

        match choice:
            case "1":
                dataset_name = input("Masukkan nama file dataset (CSV/XLSX): ").strip()
                if not os.path.exists(dataset_name):
                    print(f"❌ File {dataset_name} tidak ditemukan. Silakan sediakan file terlebih dahulu.")
                    continue
                run_standardization(dataset_name)

            case "2":
                # Pastikan standardized_data sudah ada
                data_files = sorted([f for f in os.listdir("data") if f.startswith("standardized_data")], reverse=True)
                if not data_files:
                    print("❌ Tidak ada data standardized. Jalankan menu 1 (Standardization) terlebih dahulu.")
                    continue

                # Pilih mode K
                print("\nPilih mode K:")
                print("1. Manual")
                print("2. Auto (Silhouette Score terbaik)")
                mode = input("Pilih [1-2]: ").strip()

                if mode == "1":
                    try:
                        k = int(input("Masukkan jumlah cluster (K) [2-10]: "))
                        if k < 2 or k > 10:
                            print("⚠️ K harus di antara 2–10.")
                            continue
                    except ValueError:
                        print("⚠️ Input K harus berupa angka.")
                        continue
                elif mode == "2":
                    df = pd.read_csv(os.path.join("data", data_files[0]))
                    k = auto_select_k(df, max_k=10)
                else:
                    print("⚠️ Pilihan tidak valid.")
                    continue

                # Cek apakah ada Target Audience
                master_files = sorted([f for f in os.listdir("data") if f.startswith("dataset_master")], reverse=True)
                if master_files:
                    df_master = pd.read_csv(os.path.join("data", master_files[0]))
                    if "Target Audience" in df_master.columns:
                        print("Opsi Target Audience tersedia:")
                        audiences = df_master["Target Audience"].dropna().unique()
                        for i, aud in enumerate(audiences, 1):
                            print(f"{i}. {aud}")
                        print("0. Semua")
                        try:
                            idx = int(input("Pilih Target Audience (0 untuk semua): "))
                            audience = None if idx == 0 else audiences[idx-1]
                        except Exception:
                            audience = None
                        run_clustering(k, audience)
                    else:
                        run_clustering(k)
                else:
                    print("❌ Tidak ada file master ditemukan.")

            case "3":
                print("👋 Keluar dari program.")
                break

            case _:
                print("⚠️ Pilihan tidak valid.")


if __name__ == "__main__":
    main()
