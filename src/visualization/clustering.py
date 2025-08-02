import pandas as pd


def get_mean_per_cluster(df, labels):
    """Rata-rata setiap fitur per cluster + jumlah item."""
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    mean_df = df_clustered.groupby('Cluster').mean()
    item_count = df_clustered['Cluster'].value_counts().sort_index().rename('Item Count')
    return pd.concat([item_count, mean_df], axis=1)


def get_cluster_profile(df_master, labels, num_cols, cat_cols):
    """Profil cluster gabungan (numerik = mean, kategorikal = mode)."""
    df = df_master.copy()
    df['Cluster'] = labels

    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None

    profile_num = df.groupby('Cluster')[num_cols].mean() if num_cols else pd.DataFrame()
    profile_cat = df.groupby('Cluster')[cat_cols].agg(get_mode) if cat_cols else pd.DataFrame()
    count = df['Cluster'].value_counts().sort_index().rename('Item Count')

    return pd.concat([count, profile_num, profile_cat], axis=1)
