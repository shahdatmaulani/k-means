import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_column_names(df):
    """
    rename kolom menjadi huruf kecil dan tanpa spasi
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

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


def standardize_all_numeric(df):
    """
    Standarisasi semua kolom numerik dengan Z-Score
    ❌ Kecuali kolom yang seluruh nilainya sudah 0–1
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_to_standardize = []
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        if not (min_val >= 0 and max_val <= 1):
            cols_to_standardize.append(col)

    if cols_to_standardize:
        scaler = StandardScaler()
        df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

    return df, cols_to_standardize
