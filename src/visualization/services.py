import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_RANDOM_STATE
from src.visualization.clustering import get_cluster_profile


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
    """Jalankan clustering + buat profil cluster."""
    if audience and "Target Audience" in df_master.columns:
        mask = df_master["Target Audience"] == audience
        df = df.loc[mask].reset_index(drop=True)
        df_master = df_master.loc[mask].reset_index(drop=True)

    model, labels = train_kmeans(df, k)

    num_cols = [c for c in df_master.select_dtypes(include=['int64', 'float64']).columns]
    cat_cols = [c for c in df_master.select_dtypes(include=['object', 'category']).columns]

    profile = get_cluster_profile(df_master, labels, num_cols, cat_cols)

    return {
        "labels": labels,
        "profile": profile,
        "model": model
    }
