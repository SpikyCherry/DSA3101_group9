import pandas as pd
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Load preprocessed dataset ===
data_path = 'data/processed/bank_churners_question_10_processed.csv'
df = pd.read_csv(data_path)

# === Train-test split ===
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Impute missing values (mode) ===
for col in ['Education_Level', 'Income_Category']:
    X_train[col].fillna(X_train[col].mode()[0], inplace=True)
    X_test[col].fillna(X_test[col].mode()[0], inplace=True)

# === Apply SMOTE ===
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === Train XGBoost model ===
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_resampled, y_resampled)

# === Evaluate ===
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Save model ===
model_dir = 'models/question_10'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'xgb_model.pkl')

joblib.dump(xgb_model, model_path)
print(f"Model saved to {model_path}")