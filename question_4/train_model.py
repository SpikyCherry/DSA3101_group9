import pandas as pd
import sys
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def find_best_logistic_regression_params(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> LogisticRegression:
    """
    Finds the best hyperparameters for a Logistic Regression model using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Random state for reproducibility.

    Returns:
        LogisticRegression: Best Logistic Regression model with tuned hyperparameters.
    """

    param_dist = {
        'C': loguniform(0.001, 100),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    random_search = RandomizedSearchCV(
        LogisticRegression(),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    return random_search.best_estimator_

def train_and_evaluate_logistic_regression(best_model: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Trains and evaluates a Logistic Regression model.

    Args:
        best_model (LogisticRegression): Best Logistic Regression model with tuned hyperparameters.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        tuple: (y_test, y_pred)
    """

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return y_test, y_pred

def find_best_random_forest_params(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    """
    Finds the best hyperparameters for a Random Forest model using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Best Random Forest model with tuned hyperparameters.
    """

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=20, cv=5,
        scoring='roc_auc', n_jobs=-1, random_state=random_state
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    return random_search.best_estimator_

def train_and_evaluate_random_forest(best_rf: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Trains and evaluates a Random Forest model.

    Args:
        best_rf (RandomForestClassifier): Best Random Forest model with tuned hyperparameters.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        tuple: (y_pred, y_proba, cv_scores.mean())
    """

    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f}")

    return y_pred, y_proba, cv_scores.mean()


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

encoded_path = os.path.join(parent_dir,  "data/processed/bank_customers_train_encoded.csv")
df_encoded = pd.read_csv(encoded_path)
scaled_path = os.path.join(parent_dir,  "data/processed/bank_customers_train_scaled.csv")
df_scaled = pd.read_csv(scaled_path)

X = df_encoded.drop(columns=['y'])  # Features
y = df_encoded['y']  # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Logistic Regression works better with normalized data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Find the best Logistic Regression model parameters
best_model_LR = find_best_logistic_regression_params(X_train, y_train)

# 5. Train and evaluate the model
y_test_result, y_pred = train_and_evaluate_logistic_regression(best_model_LR, X_train, y_train, X_test, y_test)

# === Save model ===
model_dir = '../models/question_4'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')

joblib.dump(best_model_LR, model_path)
print(f"Model saved to {model_path}")

best_rf = find_best_random_forest_params(X_train, y_train)
y_pred, y_proba, cv_accuracy = train_and_evaluate_random_forest(best_rf, X_train, y_train, X_test, y_test)

# === Save model ===
model_dir = '../models/question_4'
model_rf_path = os.path.join(model_dir, 'random_forest_model.pkl')

joblib.dump(best_rf, model_rf_path)
print(f"Model saved to {model_rf_path}")

