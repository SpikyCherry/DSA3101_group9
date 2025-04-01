import pandas as pd
import os
import sys

from functions import perform_clustering, plot_cluster_analysis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Load preprocessed dataset ===
data_path_encoded = 'data/processed/bank_churners_question_1_encoded.csv'
data_path_scaled = 'data/processed/bank_churners_question_1_scaled.csv'
df_encoded = pd.read_csv(data_path_encoded)
df_scaled = pd.read_csv(data_path_scaled)

# === Running Hierarchical Clustering ===
hc_data_clustered = df_encoded.copy()
  
hc_clusters = perform_clustering(df_scaled)
hc_data_clustered['Cluster'] = hc_clusters

# === Plotting Clusters ===
plot_cluster_analysis(df_scaled, hc_clusters)
