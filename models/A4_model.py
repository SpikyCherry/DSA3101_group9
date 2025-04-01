import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
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

# Example Usage (assuming X_train, y_train, X_test, y_test are already defined):
# best_rf = find_best_random_forest_params(X_train, y_train)
# y_pred, y_proba, cv_accuracy = train_and_evaluate_random_forest(best_rf, X_train, y_train, X_test, y_test)