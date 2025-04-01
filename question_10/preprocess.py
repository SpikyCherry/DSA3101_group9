import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_cleaning import drop_columns, encode_categorical, CATEGORY_ORDERS

# Define paths
input_path = 'data/raw/BankChurners.csv'
output_path = 'data/processed/bank_churners_question_10_processed.csv'

# Load data
df = pd.read_csv(input_path)

# Drop unnecessary columns
cols_to_drop = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
]
df = drop_columns(df, cols_to_drop)

# Encode categorical features
df = encode_categorical(
    df,
    label_encode_cols=['Gender', 'Attrition_Flag'],  # Includes target
    one_hot_encode_cols=['Marital_Status'],
    ordinal_encode_cols=CATEGORY_ORDERS
)

# Save processed file
os.makedirs('data/processed', exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")