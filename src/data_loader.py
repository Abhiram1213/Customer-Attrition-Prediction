import pandas as pd
import numpy as np
from typing import Tuple, List

LEAKAGE_AND_ID_COLUMNS = [
    'Customer ID',
    'Churn Status',
    'Churn Category',
    'Churn Reason',
    'Customer Churn',
    'City',
    'Zip Code'
]

NUMERIC_COLUMNS = [
    'Tenure in Months',
    'Number of Referrals',
    'Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download',
    'Monthly Charge',
    'Total Regular Charges',
    'Total Refunds',
    'Total Extra Data Charges',
    'Total Long Distance Charges',
    'Age',
    'Population',
    'CLTV',
    'Total Customer Svc Requests',
    'Product/Service Issues Reported',
    'Customer Satisfaction rate'
]

def load_data(filepath: str = 'verizonet_data.csv') -> pd.DataFrame:
    """Load raw telecom customer data from CSV."""
    if not filepath:
        raise ValueError("Filepath must be provided.")
    df = pd.read_csv(filepath)
    return df

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separates features and target variable while preventing data leakage.
    Returns:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target binary churn status (1=Churn, 0=Stay)
        numeric_cols (List[str]): List of numeric feature names
        categorical_cols (List[str]): List of categorical feature names
    """
    if 'Churn Status' not in df.columns:
        raise KeyError("Target column 'Churn Status' not found in dataframe.")
        
    y = df['Churn Status'].astype(int)
    
    # Drop IDs and post-churn leakage columns
    feature_cols = [c for c in df.columns if c not in LEAKAGE_AND_ID_COLUMNS]
    X = df[feature_cols].copy()
    
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    
    # Convert categorical columns to object/str for encoder
    for col in categorical_cols:
        X[col] = X[col].astype(object)
        
    return X, y, numeric_cols, categorical_cols
