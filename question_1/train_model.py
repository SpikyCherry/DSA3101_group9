import pandas as pd
import os
import sys
import joblib
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from functions import plot_cluster_analysis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Load preprocessed dataset ===
data_path_encoded = 'data/processed/bank_churners_question_1_encoded.csv'
data_path_scaled = 'data/processed/bank_churners_question_1_scaled.csv'
df_encoded = pd.read_csv(data_path_encoded)
df_scaled = pd.read_csv(data_path_scaled)

# === Running Hierarchical Clustering ===
hc_data_clustered = df_encoded.copy()

# Create linkage matrix
linkage_matrix = sch.linkage(df_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(8, 5))
sch.dendrogram(linkage_matrix)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
n_clusters = 5  # Set the number of clusters
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hc_clusters = hc.fit_predict(df_scaled)

# Add cluster labels to your dataframe
hc_data_clustered['Cluster'] = hc_clusters


# === Plotting Clusters ===
plot_cluster_analysis(df_scaled, hc_clusters)


# === Save model ===
model_dir = 'models/question_1'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'hc.pkl')

joblib.dump(hc, model_path)
print(f"Model saved to {model_path}")
