import os
import pandas as pd
from src.standardization.transform import (
    normalize_column_names, encode_categorical_columns, standardize_all_numeric
)
from src.standardization.io_utils import save_dataframe_with_timestamp


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
    normalize_column_names(df)

    # One-hot encoding
    df_encoded, _, _ = encode_categorical_columns(df)
    save_dataframe_with_timestamp(df_encoded, "encoded_with_numeric")

    # Standarisasi numerik (semua kolom numerik yang bukan 0-1)
    df_encoded, cols_to_standardize = standardize_all_numeric(df_encoded)

    # Simpan hasil akhir
    standardized_file, _ = save_dataframe_with_timestamp(df_encoded, "standardized_data")

    return {
        "master_file": master_file,
        "standardized_file": standardized_file,
        "timestamp": timestamp,
        "columns_standardized": cols_to_standardize,
    }
