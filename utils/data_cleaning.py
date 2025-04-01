import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

# Category orders (excluding 'Unknown')
EDUCATION_ORDER = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
INCOME_ORDER = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
CARD_ORDER = ['Blue', 'Silver', 'Gold', 'Platinum']

CATEGORY_ORDERS = {
    'Education_Level': EDUCATION_ORDER,
    'Income_Category': INCOME_ORDER,
    'Card_Category': CARD_ORDER
}

def drop_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Drops specified columns from the dataframe.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        cols_to_drop (list): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with columns removed.
    """
    return df.drop(columns=cols_to_drop, errors='ignore').copy()

def encode_categorical(
    df: pd.DataFrame,
    label_encode_cols: list = None,
    one_hot_encode_cols: list = None,
    ordinal_encode_cols: dict = None,
    binary_encode_cols: dict = None
) -> pd.DataFrame:
    """
    Encodes categorical features flexibly based on user input.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        label_encode_cols (list): Columns to label encode.
        one_hot_encode_cols (list): Columns to one-hot encode.
        ordinal_encode_cols (dict): Columns to ordinal encode with orders (e.g., {'education': ['low', 'medium', 'high']}).
        binary_encode_cols (dict): Dictionary specifying columns and their mappings for binary encoding.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    df = df.copy()

    # Label Encoding
    if label_encode_cols:
        for col in label_encode_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # One-Hot Encoding
    if one_hot_encode_cols:
        df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)

    # Ordinal Encoding
    if ordinal_encode_cols:
        ordinal_cols = []
        ordinal_orders = []
        for col, order in ordinal_encode_cols.items():  # Iterate through the dictionary
            if col in df.columns: #makes sure the column is in the dataframe.
                ordinal_cols.append(col)
                ordinal_orders.append(order)

        if ordinal_cols:
            oe = OrdinalEncoder(
                categories=ordinal_orders,
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            df[ordinal_cols] = oe.fit_transform(df[ordinal_cols].astype(str)).astype(float)
    
    # Binary Encoding
    if binary_encode_cols:
        for col, mapping in binary_encode_cols.items():
            if col in df.columns: #prevents errors if the column is not in the dataframe.
                df[col] = df[col].map(mapping)
                if np.nan in mapping.values():
                    df[col] = df[col].fillna(mapping[np.nan])

    return df

def scale_features(df: pd.DataFrame, num_features: list = None) -> pd.DataFrame:
    """
    Scales numeric features using StandardScaler. Scales all numeric features if num_features is None or empty.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        num_features (list, optional): List of numeric column names to scale. If None or empty, scales all numeric columns.

    Returns:
        pd.DataFrame: DataFrame with specified or all numeric features scaled.
    """
    df = df.copy()  # Prevent changes to original DataFrame.
    scaler = StandardScaler()
    if num_features:
        if num_features: #prevent errors if empty list is passed in.
            df[num_features] = scaler.fit_transform(df[num_features])

            return df
    else:
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns)


def drop_missing_values(df, columns_to_check):
    """
    Drops rows with missing values in the specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_check (list): A list of column names to check for missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
    """
    return df.dropna(subset=columns_to_check)