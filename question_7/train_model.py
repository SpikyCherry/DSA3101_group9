import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# --------------------------------------------
# Load Data
# --------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
banking_marketing_train_encoded = pd.read_csv('data/processed/banking_marketing_train_processed.csv', delimiter=';', quotechar='"', encoding='utf-8')
df = banking_marketing_train_encoded.copy()

# --------------------------------------------
# Duplicate 'y' column to 'conversion_binary'
# --------------------------------------------
df['conversion_binary'] = df['y']

# --------------------------------------------
# Customer Segmentation
# --------------------------------------------

# Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
df[numeric_cols] = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

# K-Means Clustering (5 Segments)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[numeric_cols])

# Define customer segment descriptions
cluster_mapping = {
    0: "Mid-High Income Males with Dependents, Strong Banking Relationship",
    1: "Young, Low-Income Females with Shortest Tenure & Low Credit",
    2: "Older, Low-Income Females with Strong Banking Relationship & High Utilisation",
    3: "Mid-Income Graduates with High Spending & Transactions",
    4: "Educated, Single Individuals with High Credit & Low Utilisation"
}
df['customer_segment'] = df['Cluster'].map(cluster_mapping)

# --------------------------------------------
# Train Models to Predict Customer Segments
# --------------------------------------------

# Select features for training
selected_cols = [
    'age', 'education', 'default', 'balance', 'housing', 'loan', 'day', 'month', 'duration',
    'campaign', 'pdays', 'previous', 'poutcome_failure', 'poutcome_other', 'poutcome_success',
    'poutcome_unknown', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
    'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed',
    'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'contact_cellular',
    'contact_telephone', 'contact_unknown'
]
df_selected = df[selected_cols].copy()
y = df['Cluster']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define and tune XGBoost model
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1, 5]
}
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
                           param_grid, cv=2, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

# Evaluate XGBoost model
y_pred_best = best_model.predict(X_test)
print(f"\nXGBoost Accuracy with Tuning: {accuracy_score(y_test, y_pred_best):.2f}")
print(classification_report(y_test, y_pred_best, zero_division=1))

# Train an ensemble model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
ensemble_model = VotingClassifier(estimators=[('xgb', best_model), ('rf', rf), ('dt', dt)], voting='hard')
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Evaluate ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
print(f"\nEnsemble Model Accuracy: {accuracy_score(y_test, y_pred_ensemble):.2f}")
print(classification_report(y_test, y_pred_ensemble, zero_division=1))

# Apply best model to full dataset
df['Predicted_Segment'] = best_model.predict(df_selected)
df['customer_segment'] = df['Predicted_Segment'].map(cluster_mapping)
df.to_csv("data/processed/Q7_banking_marketing_train_segmented.csv", index=False)
print("Customer Segmentation Completed & Full Dataset Saved!")

# --------------------------------------------
# Campaign Recommendation System
# --------------------------------------------

# Load segmented customer dataset
df = pd.read_csv("data/processed/Q7_banking_marketing_train_segmented.csv")

# Define Banking Campaign Variants
campaign_variants = ["New Account Bonus", "Spend-to-Earn Rewards", "Free Banking Services", "Loyalty Rewards Program"]
timing_variants = ["Morning", "Afternoon", "Evening"]
channel_variants = ["Email", "SMS", "Push Notification"]

# Initialize parameters for Thompson Sampling
segment_list = df["customer_segment"].unique()
historical_performance = df.groupby("customer_segment")["conversion_binary"].mean()

segment_campaign_params = {
    segment: {
        variant: {
            "alpha": max(1, historical_performance[segment] * 10),
            "beta": max(1, (1 - historical_performance[segment]) * 10)
        } for variant in campaign_variants
    }
    for segment in segment_list
}
segment_timing_params = {segment: {variant: {"alpha": 1, "beta": 1} for variant in timing_variants} for segment in segment_list}
segment_channel_params = {segment: {variant: {"alpha": 1, "beta": 1} for variant in channel_variants} for segment in segment_list}

# Define Dynamic Fatigue Score Handling
low_fatigue_threshold = df["fatigue_score"].quantile(0.33)
high_fatigue_threshold = df["fatigue_score"].quantile(0.66)

# Function to Recommend Best Campaign
def recommend_campaign(customer):
    customer_segment = customer["customer_segment"]
    fatigue_score = customer["fatigue_score"]
    conversion_rate = max(customer["conversion_rate"], 0.01)
    best_contact_time = customer["best_contact_time"]

    # Categorize fatigue levels
    if fatigue_score <= low_fatigue_threshold:
        fatigue_level = "low"
    elif fatigue_score >= high_fatigue_threshold:
        fatigue_level = "high"
    else:
        fatigue_level = "medium"

    # Adjust campaign selection based on fatigue
    adjusted_campaigns = campaign_variants.copy()
    if fatigue_level == "high":
        adjusted_campaigns.remove("Spend-to-Earn Rewards")
    elif fatigue_level == "low":
        adjusted_campaigns.append("New Account Bonus")

    # Sample from Beta distributions for campaign selection
    best_campaign = max(
        adjusted_campaigns, key=lambda x: beta.rvs(
            max(segment_campaign_params[customer_segment][x]["alpha"] + conversion_rate * 10, 1),
            max(segment_campaign_params[customer_segment][x]["beta"] + 5, 1)
        )
    )

    # Select best timing
    best_timing = best_contact_time if best_contact_time in timing_variants else max(
        timing_variants, key=lambda x: beta.rvs(
            max(segment_timing_params[customer_segment][x]["alpha"], 1),
            max(segment_timing_params[customer_segment][x]["beta"], 1)
        )
    )

    # Select best channel
    best_channel = max(channel_variants, key=lambda x: beta.rvs(
        max(segment_channel_params[customer_segment][x]["alpha"], 1),
        max(segment_channel_params[customer_segment][x]["beta"], 1)
    ))

    return best_campaign, best_timing, best_channel

# Generate Recommendations
df[["recommended_campaign", "recommended_timing", "recommended_channel"]] = df.apply(recommend_campaign, axis=1, result_type="expand")
df.to_csv("data/processed/Q7_customer_campaign_recommendations_final.csv", index=False)
print("Final Optimized Banking Campaign Recommendations Generated & Saved!")

