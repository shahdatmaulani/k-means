import pandas as pd

# ---------------------------
# 🔹 Validasi jumlah cluster K
# ---------------------------
def validate_k_input(chosen_k: int, max_k: int) -> int:
    """
    Validasi input jumlah cluster (K).
    - K tidak boleh lebih besar dari max_k
    - K minimal 2
    """
    if chosen_k > max_k:
        raise ValueError(f"❌ Nilai K={chosen_k} tidak boleh lebih besar dari max K={max_k}.")
    if chosen_k < 2:
        raise ValueError("❌ Nilai K minimal adalah 2.")
    return chosen_k


# ---------------------------------
# 🔹 Validasi & pilih Target Audience
# ---------------------------------
def validate_target_audience(
    df: pd.DataFrame, df_master: pd.DataFrame, audience: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """
    Validasi filter Target Audience:
    - Jika kolom tidak ada → return data penuh
    - Jika ada, tapi kosong → return data penuh
    - Jika filter tidak cocok → fallback ke seluruh data
    - Jika cocok → return data terfilter
    """
    if audience and "Target Audience" in df_master.columns:
        mask = df_master["Target Audience"] == audience
        if mask.sum() == 0:
            print(f"⚠️ Target Audience '{audience}' tidak ditemukan, lanjut clustering seluruh data.")
            return df, df_master, None
        else:
            return (
                df.loc[mask].reset_index(drop=True),
                df_master.loc[mask].reset_index(drop=True),
                audience,
            )
    return df, df_master, None


# ---------------------------
# 🔹 Validasi ukuran cluster
# ---------------------------
def validate_cluster_size(df: pd.DataFrame, chosen_k: int) -> int:
    """
    Validasi apakah jumlah data cukup untuk clustering dengan K tertentu.
    - K tidak boleh lebih besar dari jumlah sample.
    """
    n_samples = len(df)
    if chosen_k > n_samples:
        raise ValueError(
            f"❌ Jumlah data terlalu sedikit ({n_samples}) untuk clustering dengan K={chosen_k}."
        )
    if chosen_k == n_samples:
        print(f"⚠️ Jumlah data sama dengan K ({chosen_k}). "
              f"Hasil clustering akan trivial (setiap data 1 cluster).")
    return chosen_k
