import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def plot_elbow_silhouette(K_range, distortions, silhouettes, chosen_k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, distortions, marker='o', color='#006D77')
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.axvline(chosen_k, color='gray', linestyle='--', alpha=0.5)

    ax2.plot(K_range, silhouettes, marker='o', color='#83C5BE')
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.axvline(chosen_k, color='gray', linestyle='--', alpha=0.5)

    return fig

def plot_pca(df, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='Cluster',
        palette='Set2', s=70, edgecolor='white', alpha=0.9, ax=ax
    )
    ax.set_facecolor('#f9f9f9')
    ax.set_title("PCA Scatter Plot by Cluster")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(pca_df['Cluster'].unique()))

    return fig
