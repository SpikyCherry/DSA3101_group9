import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import data_cleaning 

output_path_encoded = '../data/processed/bank_customers_train_encoded.csv'
output_path_scaled = '../data/processed/bank_customers_train_scaled.csv'

def prepare_data(file_path):

    df = pd.read_csv(file_path)

    # Drop NA
    colums_missing_values = ['housing', 'loan', 'marital', 'education', 'job']
    df = data_cleaning.drop_missing_values(df, colums_missing_values)

    num_features = ['age', 'duration', 'campaign', 'pdays', 'previous']
    df = data_cleaning.scale_features(df, num_features)

    # Frequency Encoding for 'job'
    job_freq = df['job'].value_counts(normalize=True)
    df['job_freq'] = df['job'].map(job_freq)

    binary_mappings = {
    'default': {'no': 0, 'yes': 1, np.nan: -1},
    'housing': {'no': 0, 'yes': 1},
    'poutcome': {'failure': 0, 'success': 1, 'nonexistent': -1},
    'loan': {'no': 0, 'yes': 1},
    'y': {'no': 0, 'yes': 1}
    }

    df = data_cleaning.encode_categorical(
        df,
        label_encode_cols=['marital'],
        one_hot_encode_cols=['job', 'contact', 'month', 'day_of_week'],
        ordinal_encode_cols={'education': ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']},
        binary_encode_cols= binary_mappings
    )

    df['pdays'] = df['pdays'].replace(999, np.nan)

    df_encoded = df.copy()

    df_scaled = data_cleaning.scale_features(df_encoded)

    
    df_encoded.to_csv(output_path_encoded, index=False)
    df_scaled.to_csv(output_path_scaled, index=False)

    print(f"Encoded data saved to {output_path_encoded}")
    print(f"Scaled data saved to {output_path_scaled}")

    return df_encoded, df_scaled

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

full_path = os.path.join(parent_dir,  "data/raw/bank_customers_train.csv")
df_encoded, df_scaled = prepare_data(full_path)