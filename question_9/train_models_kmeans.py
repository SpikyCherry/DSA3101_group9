from preprocess import banking_marketing_train_encoded, banking_marketing_test_encoded
import pandas as pd
import numpy as np
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


# Determine Optimal Number of Clusters 
wcss = []  # Within-cluster sum of squares
K_range = range(1, 15)  # Start from 2 to avoid error with silhouette score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(banking_marketing_train_clustered)
    wcss.append(kmeans.inertia_)



# apply kmeans
best_k = 3 # find the best k value using elbow method

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
banking_marketing_train_clustered['Cluster'] = kmeans.fit_predict(banking_marketing_train_clustered)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(banking_marketing_train_clustered)

# Create DataFrame for PCA components with sorted contributions
# pca_df = pd.DataFrame(pca.components_, columns=banking_marketing_train_encoded.drop(columns = ['Cluster']).columns, index=['PCA1', 'PCA2'])

# Sort by the contributions to PCA1
# print(pca_df.T.sort_values(by="PCA1", ascending=False))

banking_marketing_train_clustered['PCA1'] = pca_result[:, 0]
banking_marketing_train_clustered['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x=banking_marketing_train_clustered['PCA1'], y=banking_marketing_train_clustered['PCA2'], hue=banking_marketing_train_clustered['Cluster'], palette='viridis', alpha = 0.7)
plt.title("Customer Segmentation using K-Means")
plt.show()

cluster_summary = banking_marketing_train_clustered.groupby('Cluster').mean()

# Print each row of the summary
for index, row in cluster_summary.iterrows():
    print(f"Cluster {index}:")
    print(row)
    print("\n")

banking_marketing_train_clustered_with_y = banking_marketing_train_clustered
banking_marketing_train_clustered_with_y['y'] = banking_marketing_train_encoded['y']

# Plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', hue='y', data=banking_marketing_train_clustered_with_y)
plt.title("TRAIN: Distribution of KMeans Clusters for Each Class of y")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Group the data by 'Cluster' and calculate the ratio of 1s in each cluster
cluster_ratios = banking_marketing_train_clustered_with_y.groupby('Cluster')['y'].mean()

# Find the cluster with the highest ratio of 1s
max_cluster = cluster_ratios.idxmax()
max_ratio = cluster_ratios.max()

# Print the result
print('TRAIN RESULT:')
print(f"Cluster {max_cluster} has the highest ratio of 1s with a ratio of {max_ratio:.4f}")

# cluster analysis
# Cluster 0: Appears to be a relatively stable group with moderate campaign participation and high financial balances. The higher cellular contact rate suggests good engagement with marketing efforts, though the failure rate is still significant.   -> higher success rate
# Cluster 1: Represents a more industrial, blue-collar group with lower balances and more focused telephone contact. This group has a high unknown outcome in campaigns, suggesting low engagement or response.
# Cluster 2: Features a balanced mix of job categories and high campaign participation, with 100% cellular contact indicating a more modern, tech-savvy group. This cluster is most likely to have responses marked as unknown in past campaigns.




# Validate

# Assuming 'kmeans' is the trained K-Means model from your training set
predicted_clusters = kmeans.predict(banking_marketing_test_clustered)

# Add the predicted clusters to the test data
banking_marketing_test_clustered['cluster'] = predicted_clusters

banking_marketing_test_clustered['y'] = banking_marketing_test_encoded['y']

# Plot
plt.figure(figsize=(8, 6))
sns.countplot(x='cluster', hue='y', data=banking_marketing_test_clustered)
plt.title("TEST: Distribution of KMeans Clusters for Each Class of y")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Group the data by 'Cluster' and calculate the ratio of 1s in each cluster
cluster_ratios = banking_marketing_test_clustered.groupby('cluster')['y'].mean()

# Find the cluster with the highest ratio of 1s
max_cluster = cluster_ratios.idxmax()
max_ratio = cluster_ratios.max()

# Print the result
print('TEST RESULT:')
print(f"Cluster {max_cluster} has the highest ratio of 1s with a ratio of {max_ratio:.4f}")


# Conclusion on ratio comparison: similar ratio

### Conclustion:

# Financial stability is the most important feature for deciding high-value group, and they should be priotised
# Cluster 0 customers are generally wealthier, with higher average balances and a greater likelihood of having housing loans compared to the general population. They are slightly younger and tend to be more responsive to cellular contact, though they have a higher proportion of failed outcomes and fewer responses to campaigns. These customers also have a more persistent history of contact with the bank, as indicated by higher numbers of previous contacts and longer gaps between interactions. While their job and marital status distributions are fairly similar to the broader population, Cluster 0 shows a slight tendency toward a higher proportion of single individuals.

# The test cluster 0 shows higher average balances (1639.67 vs. 1557.00) and a lower loan rate (12.13% vs. 13.59%) compared to the training cluster, suggesting a more affluent group. Both clusters exhibit similar job and marital status distributions, with strong engagement through cellular contact. However, the test cluster has a slightly higher campaign success rate (22.55% vs. 18.31%) and is less similar to the high-value segment, as indicated by a higher distance (5.93 vs. 3.61) and lower similarity (0.56 vs. 0.57).
