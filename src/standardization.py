import streamlit as st
import pandas as pd
import os
import tempfile
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from io import BytesIO
from datetime import datetime

# Constants
SESSION_KEYS = ["df_encoded", "timestamp", "current_file_key"]
PREFERRED_COLUMNS = ["price", "width", "height"]

def save_dataframe_with_timestamp(df, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", filename), index=False)
    return filename, timestamp

def reset_session_keys():
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)

def read_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx" if uploaded_file.name.endswith("xlsx") else ".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def load_dataframe(file_path, file_name):
    if file_name.endswith("csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path, engine="openpyxl")

def rename_common_columns(df):
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'price' in col_lower and '$' in col:
            rename_map[col] = 'price'
        elif 'width' in col_lower and ('inch' in col_lower or 'cm' in col_lower):
            rename_map[col] = 'width'
        elif 'height' in col_lower and ('inch' in col_lower or 'cm' in col_lower):
            rename_map[col] = 'height'
    df.rename(columns=rename_map, inplace=True)
    return rename_map

def encode_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) == 0:
        return df.copy(), [], pd.DataFrame()

    encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols)).reset_index(drop=True)
    df_numeric = df.drop(columns=categorical_cols, errors="ignore")
    df_encoded = pd.concat([df_numeric, encoded], axis=1)
    return df_encoded, list(categorical_cols), encoded

def standardize_selected_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def standardization_page():
    st.title("🧪 Data Standardization")
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"], key="file_uploader")

    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_key") != current_file_key:
            reset_session_keys()
            st.session_state["current_file_key"] = current_file_key

    if uploaded_file is not None and "df_encoded" not in st.session_state:
        if uploaded_file.size == 0:
            st.error("⚠️ File yang diupload kosong.")
            return

        temp_file_path = read_uploaded_file(uploaded_file)

        try:
            df = load_dataframe(temp_file_path, uploaded_file.name)
            
            original_filename, timestamp = save_dataframe_with_timestamp(df, "dataset_master")
            st.success(f"📁 Dataset asli berhasil disimpan sebagai '{original_filename}'")
            st.subheader("Original Data Preview")
            st.dataframe(df.head())

            rename_map = rename_common_columns(df)
            if rename_map:
                st.info(f"Renamed columns: {rename_map}")

            df_encoded, categorical_cols, encoded = encode_categorical_columns(df)
            if not encoded.empty:
                st.info(f"{len(encoded.columns)} kolom hasil one-hot encoding telah ditambahkan.")
            else:
                st.info("Tidak ditemukan kolom kategorikal untuk one-hot encoding.")

            numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cols_to_standardize = [col for col in PREFERRED_COLUMNS if col in numeric_cols]

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
                    st.error("Tidak ada kolom numerik yang dapat distandarisasi.")
                    return

            if cols_to_standardize:
                df_encoded = standardize_selected_columns(df_encoded, cols_to_standardize)
                st.success(f"Kolom yang distandarisasi: {', '.join(cols_to_standardize)}")
            else:
                st.warning("Tidak ada kolom yang dipilih untuk standardisasi.")

            st.session_state["df_encoded"] = df_encoded.copy()
            st.session_state["timestamp"] = timestamp
            st.session_state["standardized_cols"] = cols_to_standardize

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
            reset_session_keys()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    if "df_encoded" in st.session_state:
        df_encoded = st.session_state["df_encoded"]
        timestamp = st.session_state.get("timestamp")

        st.subheader("Transformed Data Preview")
        st.dataframe(df_encoded.head())
        st.caption(f"Hasil akhir: {df_encoded.shape[0]} baris × {df_encoded.shape[1]} kolom")

        if st.checkbox("Tampilkan seluruh data hasil transformasi"):
            st.dataframe(df_encoded)

        try:
            output_filename = f"standardized_data_{timestamp}.csv"
            os.makedirs("data", exist_ok=True)
            df_encoded.to_csv(os.path.join("data", output_filename), index=False)
            st.success(f"✅ Data berhasil disimpan sebagai '{output_filename}'")
        except Exception as e:
            st.warning(f"Tidak dapat menyimpan ke file lokal: {str(e)}")

        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_encoded.to_excel(writer, index=False, sheet_name="Standardized")

            st.download_button(
                label="📅 Download Standardized Data",
                data=excel_buffer.getvalue(),
                file_name=f"standardized_data_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")