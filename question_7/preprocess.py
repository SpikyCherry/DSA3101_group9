import pandas as pd
import numpy as np
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
# Duplicate the original binary 'y' column to 'conversion_binary'
# --------------------------------------------
df['conversion_binary'] = df['y']  # Duplicate 'y' column to 'conversion_binary'
# Do not touch 'conversion_binary' throughout the classifier

# --------------------------------------------
# Customer Segmentation
# --------------------------------------------

# Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

# Normalize numerical columns using StandardScaler
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
df[numeric_cols] = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), 
                                columns=numeric_cols, index=df.index)

# Perform K-Means Clustering (5 Segments)
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

# Map clusters to descriptions
df['customer_segment'] = df['Cluster'].map(cluster_mapping)

# --------------------------------------------
# Train Models to Predict Customer Segments
# --------------------------------------------

# Select relevant columns for training
selected_cols = [
    'age', 'education', 'default', 'balance', 'housing', 'loan', 'day', 'month', 'duration',
    'campaign', 'pdays', 'previous', 'poutcome_failure', 'poutcome_other', 'poutcome_success',
    'poutcome_unknown', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
    'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
    'marital_divorced', 'marital_married', 'marital_single', 'contact_cellular',
    'contact_telephone', 'contact_unknown'
]

df_selected = df[selected_cols].copy()
y = df['Cluster']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define XGBoost model with class weighting
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, learning_rate=0.1, random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1, 5]  # Adjust for imbalance
}

grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), 
                           param_grid, cv=2, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters from grid search
print(f"Best parameters: {grid_search.best_params_}")

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Model evaluation
print(f"\nXGBoost Accuracy with Tuning: {accuracy_score(y_test, y_pred_best):.2f}")
print(classification_report(y_test, y_pred_best, zero_division=1))

# --------------------------------------------
# Ensemble Model (Voting Classifier)
# --------------------------------------------

# Create an ensemble of classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
ensemble_model = VotingClassifier(estimators=[('xgb', best_model), ('rf', rf), ('dt', dt)], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train_resampled, y_train_resampled)
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate ensemble model
print(f"\nEnsemble Model Accuracy: {accuracy_score(y_test, y_pred_ensemble):.2f}")
print(classification_report(y_test, y_pred_ensemble, zero_division=1))

# --------------------------------------------
# Apply Best Model (XGBoost) to Full Dataset
# --------------------------------------------

df['Predicted_Segment'] = best_model.predict(df_selected)
df['customer_segment'] = df['Predicted_Segment'].map(cluster_mapping)

# Save the entire dataset with segmentation
df.to_csv("data/processed/Q7_banking_marketing_train_segmented.csv", index=False)

print("Customer Segmentation Completed & Full Dataset Saved!")
