import os
import glob
import tempfile
import io
import pandas as pd
import streamlit as st
from datetime import datetime
from src.config import SESSION_KEYS

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

def save_dataframe_with_timestamp(df, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", filename), index=False)
    return filename, timestamp

def load_latest_csv(folder_path, pattern="*.csv"):
    csv_files = glob.glob(os.path.join(folder_path, pattern))
    if not csv_files:
        return None, None
    latest_file = max(csv_files, key=os.path.getmtime)
    return pd.read_csv(latest_file), latest_file

def get_combined_excel_download(data_folder="data"):

    encoded_files = sorted(glob.glob(os.path.join(data_folder, "encoded_with_numeric_*.csv")), key=os.path.getmtime, reverse=True)
    standardized_files = sorted(glob.glob(os.path.join(data_folder, "standardized_data_*.csv")), key=os.path.getmtime, reverse=True)

    if not encoded_files or not standardized_files:
        return None, "File encoded atau standardized tidak ditemukan."

    try:
        df_encoded = pd.read_csv(encoded_files[0])
        df_standardized = pd.read_csv(standardized_files[0])

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_encoded.to_excel(writer, index=False, sheet_name="Encoded + Numeric")
            df_standardized.to_excel(writer, index=False, sheet_name="Standardized")

        return buffer.getvalue(), None
    except Exception as e:
        return None, str(e)
