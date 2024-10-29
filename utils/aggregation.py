# utils/aggregation.py

import pandas as pd
import streamlit as st

def aggregate_features(df, features_to_aggregate, aggregation_method, aggregated_feature_name):
    """
    Aggregates selected features in the DataFrame using the specified aggregation method.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - features_to_aggregate (list): List of feature names to aggregate.
    - aggregation_method (str): Aggregation method ('sum', 'mean', 'median').
    - aggregated_feature_name (str): Name for the new aggregated feature.

    Returns:
    - pd.DataFrame: DataFrame with the new aggregated feature added.
    """
    if not features_to_aggregate:
        st.warning("No features selected for aggregation.")
        return df

    # Validate aggregation method
    if aggregation_method not in ['sum', 'mean', 'median']:
        st.error(f"Unsupported aggregation method: {aggregation_method}")
        return df

    # Check if all features exist in the DataFrame
    missing_features = [feature for feature in features_to_aggregate if feature not in df.columns]
    if missing_features:
        st.error(f"The following features are missing in the data: {missing_features}")
        return df

    # Perform aggregation
    if aggregation_method == 'sum':
        df[aggregated_feature_name] = df[features_to_aggregate].sum(axis=1)
    elif aggregation_method == 'mean':
        df[aggregated_feature_name] = df[features_to_aggregate].mean(axis=1)
    elif aggregation_method == 'median':
        df[aggregated_feature_name] = df[features_to_aggregate].median(axis=1)
    
    st.success(f"Aggregated features {features_to_aggregate} into {aggregated_feature_name} using {aggregation_method}.")
    return df
