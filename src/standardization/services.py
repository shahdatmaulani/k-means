import os
import pandas as pd
from src.standardization.transform import (
    rename_common_columns, encode_categorical_columns, standardize_selected_columns
)
from src.standardization.io_utils import save_dataframe_with_timestamp
from src.config import PREFERRED_COLUMNS, DATA_DIR


def process_dataset(file_path: str):
    """
    Pipeline standardisasi: rename kolom, one-hot encoding, scaling, lalu simpan hasil.
    Return dict berisi nama file hasil, timestamp, dan kolom yang distandardisasi.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")

    # Load dataset
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Simpan dataset asli
    master_file, timestamp = save_dataframe_with_timestamp(df, "dataset_master")

    # Rename kolom umum
    rename_common_columns(df)

    # One-hot encoding
    df_encoded, _, _ = encode_categorical_columns(df)
    save_dataframe_with_timestamp(df_encoded, "encoded_with_numeric")

    # Standardisasi kolom numerik
    numeric_cols = df_encoded.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_to_standardize = [c for c in PREFERRED_COLUMNS if c in numeric_cols]

    if cols_to_standardize:
        df_encoded = standardize_selected_columns(df_encoded, cols_to_standardize)

    # Simpan hasil akhir
    standardized_file, _ = save_dataframe_with_timestamp(df_encoded, "standardized_data")

    return {
        "master_file": master_file,
        "standardized_file": standardized_file,
        "timestamp": timestamp,
        "columns_standardized": cols_to_standardize,
    }
