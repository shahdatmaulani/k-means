import streamlit as st
import os
import pandas as pd
from src.standardization.io_utils import (
    read_uploaded_file, reset_session_keys,
    get_combined_excel_download, update_session
)
from src.standardization.services import process_dataset


def standardization_page():
    st.title("🧪 Data Standardization")

    # Upload file CSV atau Excel
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"], key="file_uploader")

    # Reset session kalau ada file baru
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_key") != current_file_key:
            reset_session_keys()
            st.session_state["current_file_key"] = current_file_key

    # Proses dataset
    if uploaded_file is not None and "df_encoded" not in st.session_state:
        if uploaded_file.size == 0:
            st.error("⚠️ File yang diupload kosong.")
            return

        temp_file_path = read_uploaded_file(uploaded_file)

        try:
            # Jalankan pipeline standardisasi
            result = process_dataset(temp_file_path)

            # Baca hasil standardized (CSV terbaru di folder data)
            df_encoded = pd.read_csv(os.path.join("data", result["standardized_file"]))

            # Update session state
            update_session(df_encoded, result["timestamp"], result["columns_standardized"])

            st.success(f"✅ Dataset berhasil diproses & disimpan: {result['standardized_file']}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            reset_session_keys()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    # Tampilkan hasil
    if "df_encoded" in st.session_state:
        df_encoded = st.session_state["df_encoded"]

        st.subheader("📋 Data Hasil Transformasi")
        st.dataframe(df_encoded.head())
        st.caption(f"Hasil: {df_encoded.shape[0]} baris × {df_encoded.shape[1]} kolom")

        if st.checkbox("Tampilkan seluruh data hasil transformasi"):
            st.dataframe(df_encoded)

    # Download hasil
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
