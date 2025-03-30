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
    ordinal_encode_cols: list = None
) -> pd.DataFrame:
    """
    Encodes categorical features flexibly based on user input.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        label_encode_cols (list): Columns to label encode.
        one_hot_encode_cols (list): Columns to one-hot encode.
        ordinal_encode_cols (list): Columns to ordinal encode using predefined order.

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
        for col in ordinal_encode_cols:
            if col in CATEGORY_ORDERS:
                ordinal_cols.append(col)
                ordinal_orders.append(CATEGORY_ORDERS[col])
        if ordinal_cols:
            oe = OrdinalEncoder(
                categories=ordinal_orders,
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            df[ordinal_cols] = oe.fit_transform(df[ordinal_cols].astype(str)).astype(float)

    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numeric features using StandardScaler.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)
