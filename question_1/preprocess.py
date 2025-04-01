import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_cleaning import drop_columns, scale_features

# Define paths
input_path = 'data/raw/BankChurners.csv'
output_path_encoded = 'data/processed/bank_churners_question_1_encoded.csv'
output_path_scaled = 'data/processed/bank_churners_question_1_scaled.csv'

# Load data
df = pd.read_csv(input_path)

# Drop unnecessary columns
cols_to_drop = [
    "CLIENTNUM",
    "Attrition_Flag",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
]

df_encoded = drop_columns(df, cols_to_drop)

# Encode categorical features
## Apply Label Encoding
le  = LabelEncoder()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
df_encoded['Marital_Status'] = le.fit_transform(df_encoded['Marital_Status'])

## Define ordered categories
education_order = ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
income_order = ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
card_order = ['Blue', 'Silver', 'Gold', 'Platinum']

## Ensure the columns are strings
df_encoded['Education_Level'] = df_encoded['Education_Level'].astype(str)
df_encoded['Income_Category'] = df_encoded['Income_Category'].astype(str)
df_encoded['Card_Category'] = df_encoded['Card_Category'].astype(str)

## Apply Ordinal Encoding
ordinal_encoder_education = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
df_encoded['Education_Level'] = ordinal_encoder_education.fit_transform(df_encoded[['Education_Level']]).astype(float)

ordinal_encoder_income = OrdinalEncoder(categories=[income_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
df_encoded['Income_Category'] = ordinal_encoder_income.fit_transform(df_encoded[['Income_Category']]).astype(float)

ordinal_encoder_card = OrdinalEncoder(categories=[card_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
df_encoded['Card_Category'] = ordinal_encoder_card.fit_transform(df_encoded[['Card_Category']]).astype(float)

# Apply Feature Scaling
df_scaled = scale_features(df_encoded)

# Save processed file
os.makedirs('data/processed', exist_ok=True)
df_encoded.to_csv(output_path_encoded, index=False)
df_scaled.to_csv(output_path_scaled, index=False)

print(f"Encoded data saved to {output_path_encoded}")
print(f"Scaled data saved to {output_path_scaled}")
