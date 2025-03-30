import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Read in dataset
df = pd.read_csv("../data/Bankchurners.csv")

# Data Processing
# As suggested by the description in the dataset, we remove the last two columns, as they are model-encoded features.
# Drop the last two columns (Naive Bayes classifier features)
df = df.iloc[:, :-2]

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

# Define features and target variable
X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)



### Train and Test the Best Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Best Random Forest model with optimal parameters
best_rf_model = RandomForestClassifier(
    class_weight=None,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=27
)

# Train the model
best_rf_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_rf = best_rf_model.predict(X_test_scaled)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))



### Train the Model using all Data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Combine train and test data
X_full = np.vstack((X_train, X_test))
y_full = np.hstack((y_train, y_test))

# Standardize numerical features
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Train the final model with the best hyperparameters
final_rf_model = RandomForestClassifier(
    class_weight=None,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=27
)

final_rf_model.fit(X_full_scaled, y_full)

# Model is now fully trained on all available data
print("Final model trained on full dataset.")