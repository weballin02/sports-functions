# utils/correlation_utils.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_correlation(df, target_columns):
    """
    Calculates the correlation of all features with the target columns.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing features and targets.
    - target_columns (list): List of target column names.
    
    Returns:
    - pd.DataFrame: Correlation matrix.
    """
    correlation_matrix = df.corr().loc[target_columns]
    return correlation_matrix

def calculate_vif(df):
    """
    Calculates Variance Inflation Factor (VIF) for each feature.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing features.
    
    Returns:
    - pd.DataFrame: VIF scores for each feature.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def check_numeric_columns(df):
    """
    Checks and returns numeric columns in the dataframe.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to check.
    
    Returns:
    - list: List of numeric column names.
    """
    return df.select_dtypes(include=['number']).columns.tolist()

