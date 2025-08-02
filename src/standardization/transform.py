import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def rename_common_columns(df):
    """
    Rename kolom dataset agar seragam:
    - Kolom berisi 'price' dan simbol $ → 'price'
    - Kolom berisi 'width' + satuan → 'width'
    - Kolom berisi 'height' + satuan → 'height'
    Return: mapping kolom yang diubah
    """
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
    """
    One-hot encode kolom kategorikal.
    Return:
        df_encoded (DataFrame) → gabungan numerik + one-hot encoding
        categorical_cols (list) → daftar kolom kategorikal asli
        encoded (DataFrame) → hasil one-hot encoding saja
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) == 0:
        return df.copy(), [], pd.DataFrame()

    encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)

    # Normalisasi nama kolom
    encoded_columns = [col.replace(" ", "_").lower() for col in encoded_columns]

    encoded = pd.DataFrame(encoded_array, columns=encoded_columns).reset_index(drop=True)
    df_numeric = df.drop(columns=categorical_cols, errors="ignore")
    df_encoded = pd.concat([df_numeric, encoded], axis=1)

    return df_encoded, list(categorical_cols), encoded


def standardize_selected_columns(df, columns):
    """
    Standardisasi kolom numerik dengan StandardScaler.
    Return dataframe yang sudah distandardisasi.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
