import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_RANDOM_STATE
from src.visualization.clustering import get_cluster_profile
from src.visualization.validator import (
    validate_target_audience,
    validate_cluster_size,
)


def train_kmeans(df, n_clusters, random_state=DEFAULT_RANDOM_STATE):
    """Fit KMeans ke dataset, return model + labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(df)
    return model, labels


def calculate_k_metrics(df, max_k):
    """Hitung inertia & silhouette score untuk range K."""
    distortions, silhouettes = [], []
    for k in range(2, max_k + 1):
        model, labels = train_kmeans(df, k)
        distortions.append(model.inertia_)
        silhouettes.append(silhouette_score(df, labels))
    return distortions, silhouettes


def auto_select_k(df, max_k=10):
    """Cari K terbaik menggunakan Silhouette Score."""
    silhouettes = []
    for k in range(2, max_k + 1):
        model, labels = train_kmeans(df, k)
        silhouettes.append(silhouette_score(df, labels))
    best_k = np.argmax(silhouettes) + 2
    return best_k, silhouettes


def run_clustering(df, df_master, k, audience=None):
    """
    Jalankan clustering + buat profil cluster.
    """
    # Filter audience
    df_filtered, df_master_filtered, selected = validate_target_audience(df, df_master, audience)

    # Validasi ukuran cluster
    validate_cluster_size(df_filtered, k)

    # Jalankan clustering
    model, labels = train_kmeans(df_filtered, k)

    num_cols = [c for c in df_master_filtered.select_dtypes(include=['int64', 'float64']).columns]
    cat_cols = [c for c in df_master_filtered.select_dtypes(include=['object', 'category']).columns]

    profile = get_cluster_profile(df_master_filtered, labels, num_cols, cat_cols)

    return {
        "labels": labels,
        "profile": profile,
        "model": model,
        "audience": selected,
        "df_filtered": df_filtered,
    }
