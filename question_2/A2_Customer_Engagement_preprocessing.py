import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Read in dataset
df = pd.read_csv("../data/raw/BankChurners.csv")

# As suggested by the description in the dataset, we remove the last two columns, as they are model-encoded features.
# Drop the last two columns (Naive Bayes classifier features)
df = df.iloc[:, :-2]

# List of numerical columns in your dataset
numerical_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                     'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
                     'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# List of categorical columns to visualize
categorical_columns = ['Card_Category', 'Education_Level', 'Gender', 'Marital_Status', 'Income_Category']

# Ensure 'Attrition_Flag' is numeric
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 1, 'Attrited Customer': 0})

# In our model, we do one-hot encoding on gender and marital status, as there's no internal ordering among those variables.
# For unknown values, we treat them as a separate type, as for people don't want to share their marital status, they are different from other groups of people.
# For card category, income category, and education level, we use ordinal encoding. We treat unknown values as 0 to not affect the overall ordering.

# Define columns for different encoding methods
one_hot_columns = ['Gender', 'Marital_Status']
ordinal_columns = ['Card_Category', 'Income_Category', 'Education_Level']

# Define order for ordinal encoding (assign 'Unknown' as 0)
encoding_orders = {
    'Education_Level': ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
    'Card_Category': ['Unknown', 'Blue', 'Silver', 'Gold', 'Platinum'],
    'Income_Category': ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
}

# One-Hot Encoding for Gender and Marital_Status
df = pd.get_dummies(df, columns=one_hot_columns, drop_first=False)

# Ordinal Encoding for ordered categorical variables
for col, order in encoding_orders.items():
    ordinal_encoder = OrdinalEncoder(categories=[order])
    df[col] = ordinal_encoder.fit_transform(df[[col]]).astype(int)


# We can remove the first column, as the identification number is not related to our prediction.
df = df.drop(columns=['CLIENTNUM'])

# Save the cleaned dataset
path = "../data/processed/Cleaned_data_Q2.csv"
df.to_csv(path, index=False)