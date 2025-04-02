import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

# set seed
np.random.seed(369)

# Set working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import data
banking_marketing_train = pd.read_csv('data/raw/banking_marketing_train.csv', delimiter=';', quotechar='"')
banking_marketing_test = pd.read_csv('data/raw/banking_marketing_test.csv', delimiter=';', quotechar='"')

# Create copies for encoded data
banking_marketing_train_encoded = banking_marketing_train.copy()
banking_marketing_test_encoded = banking_marketing_test.copy()


# ===========================
# One-Hot Encoding (Nominal Variables)
# ===========================

nominal_columns = ['job', 'marital', 'contact', 'poutcome']
nominal_columns = [col for col in nominal_columns if col in banking_marketing_train.columns]

banking_marketing_train_encoded = pd.get_dummies(banking_marketing_train, columns=nominal_columns)
banking_marketing_test_encoded = pd.get_dummies(banking_marketing_test, columns=nominal_columns)


# ===========================
# Binary Encoding
# ===========================

binary_columns = ['default', 'housing', 'loan', 'y']
binary_columns = [col for col in binary_columns if col in banking_marketing_train.columns]

for column in binary_columns:
    banking_marketing_train_encoded[column] = banking_marketing_train[column].map({'yes': 1, 'no': 0})
    banking_marketing_test_encoded[column] = banking_marketing_test[column].map({'yes': 1, 'no': 0})


# ===========================
# Ordinal Encoding
# ===========================

ordinal_columns = ['education', 'month']
ordinal_columns = [col for col in ordinal_columns if col in banking_marketing_train.columns]

encoder = LabelEncoder()
for col in ordinal_columns:
    banking_marketing_train_encoded[col] = encoder.fit_transform(banking_marketing_train[col])
    banking_marketing_test_encoded[col] = encoder.transform(banking_marketing_test[col])

# Restore the nominal columns
for col in nominal_columns:
    banking_marketing_train_encoded[col] = banking_marketing_train[col]
    banking_marketing_test_encoded[col] = banking_marketing_test[col]


# ===========================
# Feature Engineering
# ===========================

# Interest rates from 2008/7 to 2010/12, estimated from ECB, with the help of GenAI ChatGPT
rate_table=[
    {"time":"2008/7","current": 0.75,"three_months": 3.20,"six_months": 3.60,"one_year": 3.85,"two_year": 4.05},
    {"time":"2008/12","current": 0.40,"three_months": 2.10,"six_months": 2.50,"one_year": 2.75,"two_year": 3.00},
    {"time":"2009/5","current": 0.30,"three_months": 1.20,"six_months": 1.50,"one_year": 1.70,"two_year": 2.00},
    {"time":"2009/12","current": 0.25,"three_months": 1.00,"six_months": 1.25,"one_year": 1.40,"two_year": 1.70},
    {"time":"2010/6","current": 0.20,"three_months": 0.90,"six_months": 1.10,"one_year": 1.25,"two_year": 1.55},
    {"time":"2010/12","current": 0.20,"three_months": 0.85,"six_months": 1.00,"one_year": 1.20,"two_year": 1.50}
]

# Deposit terms
deposit_products=["current","three_months","six_months","one_year","two_year"]

# Generate synthetic deposit features (amount, term, interest rate) for customers who subscribed (y=1); return NaN otherwise
def generate_deposit_features(y_value):
    deposit_amount=np.random.choice(np.arange(1000,500000+1,100)) # Deposit range
    rate_row=np.random.choice(rate_table)
    deposit_product=np.random.choice(deposit_products)
    interest_rate=rate_row[deposit_product]
    deposit_term=deposit_product
    return pd.Series([deposit_amount,deposit_term,interest_rate]) 

# deposit_amount, deposit_term, interest_rate 
# assume: actual deposit features (when y=1), campaign recommended features (when y=0)
banking_marketing_train_encoded[["deposit_amount", "term", "interest_rate"]] = banking_marketing_train_encoded['y'].apply(generate_deposit_features)
banking_marketing_test_encoded[["deposit_amount", "term", "interest_rate"]] = banking_marketing_test_encoded['y'].apply(generate_deposit_features)

# Deposit term encoding
deposit_products = ["current", "three_months", "six_months", "one_year", "two_year"]
ordinal_encoder = OrdinalEncoder(categories=[deposit_products])

banking_marketing_train_encoded['term'] = ordinal_encoder.fit_transform(banking_marketing_train_encoded[['term']]).flatten()
banking_marketing_test_encoded['term'] = ordinal_encoder.transform(banking_marketing_test_encoded[['term']]).flatten()

# conversion rate
banking_marketing_train_encoded['conversion_rate'] = banking_marketing_train_encoded['y'] / banking_marketing_train_encoded['campaign']
banking_marketing_test_encoded['conversion_rate'] = banking_marketing_test_encoded['y'] / banking_marketing_test_encoded['campaign']

# Best Contact Time Mapping
contact_time_mapping = {
    "student": "6-8pm", "retired": "12-2pm", "unemployed": "12-2pm", "housemaid": "2-4pm",
    "admin.": "4-5pm", "management": "4-5pm", "entrepreneur": "4-5pm", "blue-collar": "4-5pm",
    "self-employed": "4-5pm", "technician": "4-5pm", "services": "4-5pm", "unknown": "4-5pm"
}

banking_marketing_train_encoded["best_contact_time"] = banking_marketing_train["job"].map(contact_time_mapping)
banking_marketing_test_encoded["best_contact_time"] = banking_marketing_test["job"].map(contact_time_mapping)

# Fatigue Score Computation
banking_marketing_train_encoded["decay_factor"] = banking_marketing_train_encoded.apply(lambda row: 0.7 if (row["campaign"] + row["previous"]) > 5 else 0.5, axis=1)
banking_marketing_train_encoded["fatigue_score"] = (banking_marketing_train_encoded["campaign"] + banking_marketing_train_encoded["previous"]) * banking_marketing_train_encoded["decay_factor"]

banking_marketing_test_encoded["decay_factor"] = banking_marketing_test_encoded.apply(lambda row: 0.7 if (row["campaign"] + row["previous"]) > 5 else 0.5, axis=1)
banking_marketing_test_encoded["fatigue_score"] = (banking_marketing_test_encoded["campaign"] + banking_marketing_test_encoded["previous"]) * banking_marketing_test_encoded["decay_factor"]

# Cost Calculation
banking_marketing_train_encoded['cost'] = banking_marketing_train_encoded['duration'] * banking_marketing_train_encoded['campaign'] / 3600 * 38
banking_marketing_test_encoded['cost'] = banking_marketing_test_encoded['duration'] * banking_marketing_test_encoded['campaign'] / 3600 * 38

# Customer Lifetime Value (CLV) Calculation
# CLV = Purchase Frequency * Revenue per Purchase * Customer Lifespan 
## purchase frequency
def calc_purchase_freq(row):
    freq = 0
    if row['poutcome'] == 'success':
        freq += 1
    if row['y'] == 1:
        freq += 1
    return freq

banking_marketing_train_encoded['purchase_frequency'] = banking_marketing_train_encoded.apply(calc_purchase_freq, axis=1)
banking_marketing_test_encoded['purchase_frequency'] = banking_marketing_test_encoded.apply(calc_purchase_freq, axis=1)

## revenue per purchase, use actual deposit features (when y=1)
banking_marketing_train_encoded['revenue'] = banking_marketing_train_encoded.apply(
    lambda row: row['deposit_amount'] * row['interest_rate'] if row['y'] == 1 else 0,
    axis=1
)
banking_marketing_test_encoded['revenue'] = banking_marketing_test_encoded.apply(
    lambda row: row['deposit_amount'] * row['interest_rate'] if row['y'] == 1 else 0,
    axis=1 
)

## customer lifespan (in days)
banking_marketing_train_encoded['customer_lifespan'] = (69 - banking_marketing_train_encoded['age']) * 365
banking_marketing_test_encoded['customer_lifespan'] = (69 - banking_marketing_train_encoded['age']) * 365

## calculate CLV
banking_marketing_train_encoded['CLV'] = banking_marketing_train_encoded['purchase_frequency'] * banking_marketing_train_encoded['customer_lifespan'] * banking_marketing_train_encoded['revenue']
banking_marketing_test_encoded['CLV'] = (banking_marketing_test_encoded['purchase_frequency'] * banking_marketing_test_encoded['customer_lifespan'] * banking_marketing_test_encoded['revenue'])

# Customer Aquisition Cost (CAC) = total costs / number of new customers gained within a specific period
## new_customer
def is_new_customer(row):
    return row['poutcome'] != 'success' and row['y'] == 1

banking_marketing_train_encoded['new_customer'] = banking_marketing_train_encoded.apply(is_new_customer, axis=1)
banking_marketing_test_encoded['new_customer'] = banking_marketing_test_encoded.apply(is_new_customer, axis=1)

## compute CAC at the group level (by contact type)
cac_by_contact_train = (
    banking_marketing_train_encoded
    .groupby('contact')
    .apply(lambda d: d['cost'].sum() / d['new_customer'].sum() if d['new_customer'].sum() > 0 else float('nan'))
    .reset_index(name='CAC')
)

cac_by_contact_test = (
    banking_marketing_test_encoded
    .groupby('contact')
    .apply(lambda d: d['cost'].sum() / d['new_customer'].sum() if d['new_customer'].sum() > 0 else float('nan'))
    .reset_index(name='CAC')
)

## assign CAC to each row
banking_marketing_train_encoded = banking_marketing_train_encoded.merge(cac_by_contact_train, on='contact', how='left')
banking_marketing_test_encoded = banking_marketing_test_encoded.merge(cac_by_contact_test, on='contact', how='left')

# ROI = CLV/CAC
banking_marketing_train_encoded['ROI'] = banking_marketing_train_encoded['CLV']  / banking_marketing_train_encoded['CAC']
banking_marketing_test_encoded['ROI'] = banking_marketing_test_encoded['CLV']  / banking_marketing_test_encoded['CAC']

# Label encode nominal_columns
encoder = LabelEncoder()
for col in nominal_columns:
    banking_marketing_train_encoded[col] = encoder.fit_transform(banking_marketing_train[col])
    banking_marketing_test_encoded[col] = encoder.transform(banking_marketing_test[col])

# export data for EDA
banking_marketing_train_encoded_eda = banking_marketing_train_encoded.copy()
banking_marketing_test_encoded_eda = banking_marketing_test_encoded.copy()

# ===========================
# Data Normalization
# ===========================

# Numeric columns to standardize
numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'deposit_amount']
numeric_columns = [col for col in numeric_columns if col in banking_marketing_train.columns]

scaler = StandardScaler()
for col in numeric_columns:
    banking_marketing_train_encoded[col] = scaler.fit_transform(banking_marketing_train[col].values.reshape(-1, 1)).flatten()
    banking_marketing_test_encoded[col] = scaler.transform(banking_marketing_test[col].values.reshape(-1, 1)).flatten()

numeric_columns = ['deposit_amount']
numeric_columns = [col for col in numeric_columns if col in banking_marketing_train.columns]

banking_marketing_train_encoded['deposit_amount'] = scaler.fit_transform(banking_marketing_train_encoded['deposit_amount'].values.reshape(-1, 1)).flatten()
banking_marketing_test_encoded['deposit_amount'] = scaler.transform(banking_marketing_test_encoded['deposit_amount'].values.reshape(-1, 1)).flatten()

banking_marketing_train_encoded['fatigue_score'] = scaler.fit_transform(banking_marketing_train_encoded['deposit_amount'].values.reshape(-1, 1)).flatten()
banking_marketing_test_encoded['fatigue_score'] = scaler.transform(banking_marketing_test_encoded['deposit_amount'].values.reshape(-1, 1)).flatten()


# ===========================
# Save Processed Data
# ===========================
banking_marketing_train_encoded.to_csv('data/processed/banking_marketing_train_processed.csv', index=False, sep=';', quotechar='"', encoding='utf-8')
banking_marketing_test_encoded.to_csv('data/processed/banking_marketing_test_processed.csv', index=False, sep=';', quotechar='"', encoding='utf-8')
banking_marketing_train_encoded_eda.to_csv('data/processed/banking_marketing_train_encoded_eda.csv', index=False, sep=';', quotechar='"', encoding='utf-8')

