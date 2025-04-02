# -*- coding: utf-8 -*-
"""
# Importing Packages & Files
"""
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

"""# Preparing Data """

def prepare_data(file_path):
    """
    Loads, cleans, encodes, and scales the dataset for clustering.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Scaled dataset ready for clustering.
        pd.DataFrame: Cleaned dataset before scaling (for reference).
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    '''df.drop([
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ], axis=1, inplace=True)'''

    df_full = df.copy()

    # Select relevant features
    df_subset = df[['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 
                    'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']].copy()

    # Label Encoding for Gender & Marital_Status
    label_encoder = LabelEncoder()
    df_subset['Gender'] = label_encoder.fit_transform(df_subset['Gender'])
    df_subset['Marital_Status'] = label_encoder.fit_transform(df_subset['Marital_Status'])

    # Define ordered categories for ordinal encoding
    education_order = ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
    income_order = ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
    card_order = ["Blue", "Silver", "Gold", "Platinum"]

    # Convert to string to avoid errors
    df_subset['Education_Level'] = df_subset['Education_Level'].astype(str)
    df_subset['Income_Category'] = df_subset['Income_Category'].astype(str)
    df_subset['Card_Category'] = df_subset['Card_Category'].astype(str)

    # Apply ordinal encoding
    ordinal_encoder_education = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
    df_subset['Education_Level'] = ordinal_encoder_education.fit_transform(df_subset[['Education_Level']]).astype(float)

    ordinal_encoder_income = OrdinalEncoder(categories=[income_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
    df_subset['Income_Category'] = ordinal_encoder_income.fit_transform(df_subset[['Income_Category']]).astype(float)

    ordinal_encoder_card = OrdinalEncoder(categories=[card_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
    df_subset['Card_Category'] = ordinal_encoder_card.fit_transform(df_subset[['Card_Category']]).astype(float)

    # Save cleaned dataset before scaling
    df_cleaned = df_subset.copy()

    # Feature Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned)
    df_scaled = pd.DataFrame(df_scaled, columns=df_cleaned.columns)

    return df_full, df_scaled, df_cleaned


# Load and preprocess the data
df_full, df_scaled, df_cleaned = prepare_data('data/raw/BankChurners.csv')
hc_data_clustered = df_cleaned.copy()


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
    #linkage_matrix = sch.linkage(data, method='ward')

    # Apply Agglomerative Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters = hc.fit_predict(data)

    return clusters

df_full, df_scaled, df_cleaned=prepare_data('data/raw/BankChurners.csv')
cluster_labels=perform_clustering(df_scaled)
df_scaled['Cluster'] = cluster_labels
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/BankChurners_clustered.csv'
df_scaled.to_csv(output_path, index=False)