import streamlit as st
import pandas as pd
import os
import tempfile
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from io import BytesIO
from datetime import datetime

def standardization_page():
    st.title("🧪 Data Standardization")

    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"], key="file_uploader")

    # Reset session state if new file is uploaded
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_key") != current_file_key:
            # Clear previous data when new file is uploaded
            for key in ["df_encoded", "timestamp", "current_file_key"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["current_file_key"] = current_file_key

    if uploaded_file is not None and "df_encoded" not in st.session_state:
        if uploaded_file.size == 0:
            st.error("⚠️ File yang diupload kosong.")
            return

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx" if uploaded_file.name.endswith("xlsx") else ".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_file_path = tmp.name

            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(temp_file_path)
            else:
                df = pd.read_excel(temp_file_path, engine="openpyxl")

            st.subheader("Original Data Preview")
            st.dataframe(df.head())

            # More flexible column renaming (case-insensitive)
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'price' in col_lower and '$' in col:
                    rename_map[col] = 'price'
                elif 'width' in col_lower and ('inch' in col_lower or 'cm' in col_lower):
                    rename_map[col] = 'width'  
                elif 'height' in col_lower and ('inch' in col_lower or 'cm' in col_lower):
                    rename_map[col] = 'height'
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
                st.info(f"Renamed columns: {rename_map}")

            # Handle categorical encoding
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
                encoded_array = encoder.fit_transform(df[categorical_cols])
                encoded = pd.DataFrame(
                    encoded_array,
                    columns=encoder.get_feature_names_out(categorical_cols)
                ).reset_index(drop=True)  # Ensure clean index
                st.info(f"{len(encoded.columns)} kolom hasil one-hot encoding telah ditambahkan.")
            else:
                st.info("Tidak ditemukan kolom kategorikal untuk one-hot encoding.")
                encoded = pd.DataFrame()  # Empty DataFrame

            # Combine numerical and encoded data
            df_numeric = df.drop(columns=categorical_cols, errors="ignore")
            df_encoded = pd.concat([df_numeric, encoded], axis=1)

            # Dynamic column selection for standardization
            numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Prefer specific columns if they exist, otherwise let user choose
            preferred_cols = ["price", "width", "height"]
            cols_to_standardize = [col for col in preferred_cols if col in numeric_cols]
            
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
                scaler = StandardScaler()
                df_encoded[cols_to_standardize] = scaler.fit_transform(df_encoded[cols_to_standardize])
                st.success(f"Kolom yang distandarisasi: {', '.join(cols_to_standardize)}")
            else:
                st.warning("Tidak ada kolom yang dipilih untuk standardisasi.")

            st.session_state["df_encoded"] = df_encoded.copy()
            st.session_state["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["standardized_cols"] = cols_to_standardize

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
            # Clear session state on error
            for key in ["df_encoded", "timestamp", "current_file_key"]:
                if key in st.session_state:
                    del st.session_state[key]

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    # Display results if data exists in session state
    if "df_encoded" in st.session_state:
        df_encoded = st.session_state["df_encoded"]
        timestamp = st.session_state.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        st.subheader("Transformed Data Preview")
        st.dataframe(df_encoded.head())
        st.caption(f"Hasil akhir: {df_encoded.shape[0]} baris × {df_encoded.shape[1]} kolom")

        if st.checkbox("Tampilkan seluruh data hasil transformasi"):
            st.dataframe(df_encoded)

        # Save and download options
        try:
            os.makedirs("data", exist_ok=True)
            output_path = os.path.join("data", f"standardized_data_{timestamp}.csv")
            df_encoded.to_csv(output_path, index=False)
            st.success(f"✅ Data berhasil disimpan ke: `{output_path}`")
        except Exception as e:
            st.warning(f"Tidak dapat menyimpan ke file lokal: {str(e)}")

        # Download button
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_encoded.to_excel(writer, index=False, sheet_name="Standardized")
            
            st.download_button(
                label="📥 Download Standardized Data",
                data=excel_buffer.getvalue(),
                file_name=f"standardized_data_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")