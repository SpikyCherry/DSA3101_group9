from DSA3101_group9.util.subB_data_preprocessing import banking_marketing_train_encoded, banking_marketing_test_encoded

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

banking_marketing_train_clustered = banking_marketing_train_encoded
banking_marketing_test_clustered = banking_marketing_test_encoded

demographic_features = ['age', 'education','job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single']
financial_features = ['default', 'balance', 'housing', 'loan',]
behaviour_features = ['campaign','contact_cellular', 'contact_telephone', 'contact_unknown',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown', 'pdays', 'previous']
columns_for_clustering = demographic_features + financial_features + behaviour_features


# Extract the relevant columns
banking_marketing_train_clustered = banking_marketing_train_clustered[columns_for_clustering]
banking_marketing_test_clustered = banking_marketing_test_clustered[columns_for_clustering]


# # Standardize the data 
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(banking_marketing_train_clustered)
# data_scaled_test = scaler.fit_transform(banking_marketing_test_clustered)


# apply kmeans
best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
banking_marketing_train_clustered['Cluster'] = kmeans.fit_predict(banking_marketing_train_clustered)

pca = PCA(n_components=2)
# pca_result = pca.fit_transform(data_scaled)
pca_result = pca.fit_transform(banking_marketing_train_clustered)

# Create DataFrame for PCA components with sorted contributions
# pca_df = pd.DataFrame(pca.components_, columns=banking_marketing_train_encoded.drop(columns = ['Cluster']).columns, index=['PCA1', 'PCA2'])

# Sort by the contributions to PCA1
# print(pca_df.T.sort_values(by="PCA1", ascending=False))

banking_marketing_train_clustered['PCA1'] = pca_result[:, 0]
banking_marketing_train_clustered['PCA2'] = pca_result[:, 1]

cluster_summary = banking_marketing_train_clustered.groupby('Cluster').mean()

# Print each row of the summary
for index, row in cluster_summary.iterrows():
    print(f"Cluster {index}:")
    print(row)
    print("\n")

banking_marketing_train_clustered_with_y = banking_marketing_train_clustered
banking_marketing_train_clustered_with_y['y'] = banking_marketing_train_encoded['y']

# Assuming 'Cluster' column is in your dataframe with KMeans results
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', hue='y', data=banking_marketing_train_clustered_with_y)
plt.title("Distribution of KMeans Clusters for Each Class of y")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Group the data by 'Cluster' and calculate the ratio of 1s in each cluster
cluster_ratios = banking_marketing_train_clustered_with_y.groupby('Cluster')['y'].mean()

# Find the cluster with the highest ratio of 1s
max_cluster = cluster_ratios.idxmax()
max_ratio = cluster_ratios.max()

# Print the result
print(f"Cluster {max_cluster} has the highest ratio of 1s with a ratio of {max_ratio:.4f}")

# cluster analysis
# Cluster 0: Appears to be a relatively stable group with moderate campaign participation and high financial balances. The higher cellular contact rate suggests good engagement with marketing efforts, though the failure rate is still significant.   -> higher success rate
# Cluster 1: Represents a more industrial, blue-collar group with lower balances and more focused telephone contact. This group has a high unknown outcome in campaigns, suggesting low engagement or response.
# Cluster 2: Features a balanced mix of job categories and high campaign participation, with 100% cellular contact indicating a more modern, tech-savvy group. This cluster is most likely to have responses marked as unknown in past campaigns.


# validate

# Assuming 'kmeans' is the trained K-Means model from your training set
predicted_clusters = kmeans.predict(banking_marketing_test_clustered)

# Add the predicted clusters to the test data
banking_marketing_test_clustered['cluster'] = predicted_clusters

# Assuming 'Cluster' column is in your dataframe with KMeans results
plt.figure(figsize=(8, 6))
sns.countplot(x='cluster', hue='y', data=banking_marketing_test_clustered)
plt.title("Distribution of KMeans Clusters for Each Class of y")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Group the data by 'Cluster' and calculate the ratio of 1s in each cluster
cluster_ratios = banking_marketing_test_clustered.groupby('cluster')['y'].mean()

# Find the cluster with the highest ratio of 1s
max_cluster = cluster_ratios.idxmax()
max_ratio = cluster_ratios.max()

# Print the result
print(f"Cluster {max_cluster} has the highest ratio of 1s with a ratio of {max_ratio:.4f}")


# conclusion: similar ratio