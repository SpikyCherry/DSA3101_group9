from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import sys
import joblib

# import data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the processed datasets
banking_marketing_train_encoded = pd.read_csv('data/processed/banking_marketing_train_processed.csv', delimiter=';', quotechar='"', encoding='utf-8')
banking_marketing_test_encoded = pd.read_csv('data/processed/banking_marketing_test_processed.csv', delimiter=';', quotechar='"', encoding='utf-8')

# Assign to variables for convenience
train, test = banking_marketing_train_encoded, banking_marketing_test_encoded

# Display the shape of training and test sets before filtering
print("Train shape before filtering:", train.shape)
print("Test shape before filtering:", test.shape)

# Check for missing values & invalid values (less than -1) in ROI
print("ROI null values in train:", train["ROI"].isnull().sum())
print("ROI < -1 in train:", (train["ROI"] < -1).sum())

# Define feature columns used for training the ROI prediction model
features = ['age', 'education', 'default', 'balance', 'housing', 'loan', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'contact_cellular', 'contact_unknown', 'poutcome_success',
       'poutcome_unknown', 'y', 'conversion_rate', 'interest_rate']

# Extract features (X) and target variable (y = ROI)
X = train[features]
y = train["ROI"]

# Filter out invalid ROI values (must be >= -1)
train = train[train["ROI"] >= -1].copy()
test = test[test["ROI"] >= -1].copy()

# ROI range from -1 to 1.195728e+09 
# Apply log1p to smooth extreme ROI values (highly skewed), reduce large outlier impact
train["ROI_log"] = np.log1p(train["ROI"])
test["ROI_log"] = np.log1p(test["ROI"])

# Remove rows with non-finite log(ROI) values (e.g., NaN or ±inf)
train = train[np.isfinite(train["ROI_log"])].copy()
test = test[np.isfinite(test["ROI_log"])].copy()

# Prepare training and test data (fill missing values with 0)
X_train = train[features].fillna(0)
y_train = train["ROI_log"]
X_test = test[features].fillna(0)
y_test = test["ROI_log"]

# Train a linear regression model using log-transformed ROI
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and inverse the log transform to recover original ROI scale
y_pred_log = lr.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Define a pipeline: Standardize + Ridge Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Define parameter grid: alpha is the regularization strength
param_grid = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 100, 200]
}

# Set up grid search with 5-fold cross-validation
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# the best model
best_model = grid.best_estimator_

# Create model directory
model_dir = 'models/question_8'
os.makedirs(model_dir, exist_ok=True)

# Define path to save the model
model_path = os.path.join(model_dir, 'ridge_model.pkl')

# Save the best Ridge regression model
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Evaluate
y_pred = best_model.predict(X_test)

print("Best alpha:", grid.best_params_['ridge__alpha'])
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R²:", r2_score(y_test, y_pred))


