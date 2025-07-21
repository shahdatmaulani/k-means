import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_k_metrics(df, max_k):
    distortions, silhouettes = [], []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42).fit(df)
        distortions.append(model.inertia_)
        silhouettes.append(silhouette_score(df, model.labels_))
    return distortions, silhouettes

def get_mean_per_cluster(df, labels):
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    mean_df = df_clustered.groupby('Cluster').mean()
    item_count = df_clustered['Cluster'].value_counts().sort_index().rename('Item Count')
    return pd.concat([item_count, mean_df], axis=1)

def get_cluster_profile(df_master, labels, num_cols, cat_cols):
    df = df_master.copy()
    df['Cluster'] = labels

    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None

    profile_num = df.groupby('Cluster')[num_cols].mean()
    profile_cat = df.groupby('Cluster')[cat_cols].agg(get_mode)
    count = df['Cluster'].value_counts().sort_index().rename('Item Count')
    return pd.concat([count, profile_num, profile_cat], axis=1)
