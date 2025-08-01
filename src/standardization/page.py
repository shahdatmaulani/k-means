import streamlit as st
from io import BytesIO
import os
import pandas as pd
from src.standardization.io_utils import (
    read_uploaded_file, load_dataframe,
    save_dataframe_with_timestamp, reset_session_keys,
    get_combined_excel_download
)
from src.standardization.transform import (
    rename_common_columns, encode_categorical_columns,
    standardize_selected_columns
)
from src.config import PREFERRED_COLUMNS

def standardization_page():
    st.title("🧪 Data Standardization")

    # Upload file CSV atau Excel
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"], key="file_uploader")

    # Cek apakah file baru diupload, jika ya, reset session state
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_key") != current_file_key:
            reset_session_keys()
            st.session_state["current_file_key"] = current_file_key

    # Jika file berhasil diupload dan belum pernah diproses
    if uploaded_file is not None and "df_encoded" not in st.session_state:
        if uploaded_file.size == 0:
            st.error("⚠️ File yang diupload kosong.")
            return

        # Simpan file upload ke file sementara
        temp_file_path = read_uploaded_file(uploaded_file)

        try:
            # Load ke dalam dataframe
            df = load_dataframe(temp_file_path, uploaded_file.name)
            
            # Simpan dataset asli ke file lokal dengan timestamp
            original_filename, timestamp = save_dataframe_with_timestamp(df, "dataset_master")
            st.success(f"✅ Dataset asli berhasil disimpan sebagai '{original_filename}'")
            
            # Tampilkan preview data
            st.subheader("Original Data Preview")
            st.dataframe(df.head())

            # Rename kolom jika ada yang cocok dengan pattern 'price', 'width', 'height'
            rename_map = rename_common_columns(df)
            #if rename_map:
            #    st.info(f"Renamed columns: {rename_map}")

            # One-hot encoding kolom kategorikal
            df_encoded, categorical_cols, encoded = encode_categorical_columns(df)
            categorical_cols = categorical_cols if categorical_cols is not None else []
            if not encoded.empty:
                #st.info(f"{len(encoded.columns)} kolom hasil one-hot encoding telah ditambahkan.")
                # Simpan gabungan kolom numerik dan hasil one-hot encoding (sebelum standardisasi)
                onehot_full_filename, _ = save_dataframe_with_timestamp(df_encoded, "encoded_with_numeric")
                st.success(f"✅ Data gabungan numerik + one-hot encoding berhasil disimpan sebagai '{onehot_full_filename}'")
            else:
                st.info("Tidak ditemukan kolom kategorikal untuk one-hot encoding.")

            # Deteksi kolom numerik yang bisa distandarisasi
            numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cols_to_standardize = [col for col in PREFERRED_COLUMNS if col in numeric_cols]

            # Jika kolom preferensi tidak ditemukan, beri pilihan manual
            if not cols_to_standardize:
                st.warning("Kolom 'price', 'width', 'height' tidak ditemukan.")
                if numeric_cols:
                    st.subheader("Pilih kolom untuk standardisasi:")
                    cols_to_standardize = st.multiselect(
                        "Kolom numerik yang tersedia:",
                        options=numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )
                else:
                    st.info("⚠️ Tidak ada kolom numerik ditemukan. Proses standardisasi dilewati, hanya hasil one-hot encoding yang disimpan.")
                    cols_to_standardize = []

            # Lakukan standardisasi pada kolom terpilih
            if cols_to_standardize:
                df_encoded = standardize_selected_columns(df_encoded, cols_to_standardize)
                #st.success(f"Kolom yang distandarisasi: {', '.join(cols_to_standardize)}")
            else:
                st.warning("Tidak ada kolom yang dipilih untuk standardisasi.")

            # Simpan ke session state untuk digunakan kembali
            st.session_state["df_encoded"] = df_encoded.copy()
            st.session_state["timestamp"] = timestamp
            st.session_state["standardized_cols"] = cols_to_standardize

        except Exception as e:
            # Tangani error saat proses
            st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
            reset_session_keys()
        finally:
            # Hapus file sementara
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    # Jika sudah ada hasil di session_state, tampilkan
    if "df_encoded" in st.session_state:
        df_encoded = st.session_state["df_encoded"]
        timestamp = st.session_state.get("timestamp")

        # Tampilkan hasil transformasi
        st.subheader("Transformed Data Preview")
        st.dataframe(df_encoded.head())
        st.caption(f"Hasil akhir: {df_encoded.shape[0]} baris × {df_encoded.shape[1]} kolom")

        # Opsi tampilkan seluruh data
        if st.checkbox("Tampilkan seluruh data hasil transformasi"):
            st.dataframe(df_encoded)

        # Simpan hasil transformasi ke CSV lokal
        try:
            output_filename = f"standardized_data_{timestamp}.csv"
            os.makedirs("data", exist_ok=True)
            df_encoded.to_csv(os.path.join("data", output_filename), index=False)
            st.success(f"✅ Data berhasil disimpan sebagai '{output_filename}'")
        except Exception as e:
            st.warning(f"Tidak dapat menyimpan ke file lokal: {str(e)}")

    st.subheader("📥 Download Hasil")

    excel_bytes, error = get_combined_excel_download()
    if excel_bytes:
        st.download_button(
            label="⬇️ Download Excel: Encoded + Standardized",
            data=excel_bytes,
            file_name="combined_encoded_standardized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif error:
        st.info(f"ℹ️ {error}")