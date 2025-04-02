import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

def plot_top_features(model: BaseEstimator, X: pd.DataFrame, k: int = 15) -> None:
    """
    Plots the top k most important features from a trained model.

    Args:
        model (BaseEstimator): Trained model (RandomForestClassifier, LogisticRegression, etc.).
        X (pd.DataFrame): DataFrame containing the features used to train the model.
        k (int): Number of top features to plot.
    """

    if hasattr(model, 'feature_importances_'):  # Random Forest and similar
        importances = pd.Series(model.feature_importances_, index=X.columns)
    elif hasattr(model, 'coef_'):  # Logistic Regression
        importances = pd.Series(model.coef_[0], index=X.columns)
    else:
        print("Error: The provided model does not have feature importances or coefficients.")
        return

    top_features = importances.abs().sort_values(ascending=False).head(k)

    plt.figure(figsize=(10, 6))
    top_features.sort_values(ascending=True).plot(kind='barh')
    plt.title(f'Top {k} Important Features')
    plt.xlabel('Feature Importance/Coefficient Magnitude')
    plt.show()