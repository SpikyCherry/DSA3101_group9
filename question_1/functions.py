import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# === Functions ===
## === Perform Clustering ===
def perform_clustering(data, n_clusters=5):
    """
    Performs hierarchical clustering and returns the clustered data.

    Parameters:
        data (pd.DataFrame or np.array): Scaled dataset for clustering.
        n_clusters (int): Number of clusters to create.

    Returns:
        np.array: Cluster labels assigned to each data point.
        
    """
    # Create linkage matrix
    linkage_matrix = sch.linkage(data, method='ward')

    # Plot Dendrogram
    plt.figure(figsize=(8, 5))
    sch.dendrogram(linkage_matrix)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

    # Apply Agglomerative Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters = hc.fit_predict(data)

    return clusters
  
## === Plotting Clusters Distribution ===
def plot_cluster_analysis(data, clusters):
    """
    Plots 2D and 3D PCA scatter plots for the clustered data.
    """
    # Reduce dimensions to 2 for visualization
    pca_2d = PCA(n_components=2)
    df_pca_2d = pca_2d.fit_transform(data)

    # Reduce dimensions to 3 for visualization
    pca_3d = PCA(n_components=3)
    df_pca_3d = pca_3d.fit_transform(data)

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={'projection': None})

    # 2D Scatter Plot
    axes[0].scatter(df_pca_2d[:, 0], df_pca_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Hierarchical Clustering (2D PCA)")
    axes[0].grid(True)

    # 3D Scatter Plot
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c=clusters, cmap='viridis', alpha=0.7)
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.set_title("Hierarchical Clustering (3D PCA)")

    plt.tight_layout()
    plt.show()
