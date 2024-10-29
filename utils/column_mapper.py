# utils/column_mapper.py

import streamlit as st
import pandas as pd
import json
import os

# Define the directory to save mappings
MAPPING_DIR = "mappings"

# Ensure the mapping directory exists
if not os.path.exists(MAPPING_DIR):
    os.makedirs(MAPPING_DIR)

# Predefined sport column mappings
PREDEFINED_SPORT_COLUMN_MAPPING = {
    'NFL': {
        'completions': 'completions',
        'attempts': 'attempts',
        'passing_yards': 'passing_yards',
        'passing_tds': 'passing_tds',
        'interceptions': 'interceptions',
        # ... [other NFL mappings]
    },
    'NBA': {
        'points': 'points',
        'rebounds': 'rebounds',
        'assists': 'assists',
        'steals': 'steals',
        'blocks': 'blocks',
        'turnovers': 'turnovers',
        # ... [other NBA mappings]
    }
    # Add more sports and their mappings as needed
}

def get_available_columns(simulation_data):
    """
    Returns a list of available columns in the simulation data.
    """
    return simulation_data.columns.tolist()

def interactive_column_mapping(sport, model_features, simulation_data):
    """
    Provides an interactive interface for users to map model features to simulation data columns,
    including the option to aggregate multiple columns into a single feature.
    
    Parameters:
    - sport (str): Selected sport.
    - model_features (list): List of feature names used in the model.
    - simulation_data (pd.DataFrame): DataFrame containing simulation data.
    
    Returns:
    - dict: Mapping of model features to simulation data columns or aggregation details.
    """
    st.sidebar.header("Feature Mapping")
    st.sidebar.write("Map each model feature to a column in your simulation data or define an aggregation.")
    
    available_columns = get_available_columns(simulation_data)
    
    # Initialize mapping dictionary
    mapping = {}
    
    for feature in model_features:
        st.sidebar.write(f"**Model Feature:** {feature}")
        
        # Option to choose mapping type
        mapping_type = st.sidebar.radio(
            f"Mapping Type for '{feature}'",
            options=["Single Column", "Aggregation"],
            key=f"map_type_{feature}"
        )
        
        if mapping_type == "Single Column":
            # Pre-fill with predefined mapping if available
            predefined_mapping = PREDEFINED_SPORT_COLUMN_MAPPING.get(sport, {}).get(feature, '')
            
            # Dropdown to select corresponding simulation data column
            mapped_column = st.sidebar.selectbox(
                f"Select column for '{feature}'",
                options=["None"] + available_columns,
                index=1 if predefined_mapping in available_columns else 0,
                key=f"map_single_{feature}"
            )
            
            if mapped_column != "None":
                mapping[feature] = {
                    'type': 'single',
                    'column': mapped_column
                }
            else:
                mapping[feature] = {
                    'type': 'single',
                    'column': None
                }
        
        elif mapping_type == "Aggregation":
            # Multiselect to choose multiple columns
            aggregated_columns = st.sidebar.multiselect(
                f"Select columns to aggregate for '{feature}'",
                options=available_columns,
                key=f"map_agg_columns_{feature}"
            )
            
            if aggregated_columns:
                # Select aggregation method
                aggregation_method = st.sidebar.selectbox(
                    f"Select aggregation method for '{feature}'",
                    options=["sum", "mean", "median"],
                    key=f"map_agg_method_{feature}"
                )
                
                # Input for the name of the aggregated feature
                aggregated_feature_name = st.sidebar.text_input(
                    f"Name for aggregated feature of '{feature}'",
                    value=f"{feature}_aggregated",
                    key=f"map_agg_name_{feature}"
                )
                
                if aggregated_feature_name:
                    mapping[feature] = {
                        'type': 'aggregation',
                        'columns': aggregated_columns,
                        'method': aggregation_method,
                        'aggregated_feature_name': aggregated_feature_name
                    }
                else:
                    mapping[feature] = {
                        'type': 'aggregation',
                        'columns': aggregated_columns,
                        'method': aggregation_method,
                        'aggregated_feature_name': None
                    }
            else:
                mapping[feature] = {
                    'type': 'aggregation',
                    'columns': [],
                    'method': None,
                    'aggregated_feature_name': None
                }
    
    # Identify missing mappings
    missing_features = []
    for feat, details in mapping.items():
        if details['type'] == 'single' and details['column'] is None:
            missing_features.append(feat)
        elif details['type'] == 'aggregation':
            if not details['columns'] or not details['method'] or not details['aggregated_feature_name']:
                missing_features.append(feat)
    
    if missing_features:
        st.sidebar.error(f"The following model features are not fully mapped: {missing_features}")
    else:
        st.sidebar.success("All model features are fully mapped.")
    
    return mapping

def apply_column_mapping(mapping, simulation_data):
    """
    Applies the column mapping to the simulation data, performing aggregations as needed.
    
    Parameters:
    - mapping (dict): Mapping of model features to simulation data columns or aggregation details.
    - simulation_data (pd.DataFrame): DataFrame containing simulation data.
    
    Returns:
    - pd.DataFrame: DataFrame with mapped and aggregated features aligned to model features.
    """
    mapped_columns = {}
    missing_features = []
    
    for feature, details in mapping.items():
        if details['type'] == 'single':
            column = details['column']
            if column and column in simulation_data.columns:
                mapped_columns[feature] = simulation_data[column]
            else:
                missing_features.append(feature)
        elif details['type'] == 'aggregation':
            columns = details['columns']
            method = details['method']
            agg_name = details['aggregated_feature_name']
            if all(col in simulation_data.columns for col in columns) and agg_name:
                if method == 'sum':
                    mapped_columns[feature] = simulation_data[columns].sum(axis=1)
                elif method == 'mean':
                    mapped_columns[feature] = simulation_data[columns].mean(axis=1)
                elif method == 'median':
                    mapped_columns[feature] = simulation_data[columns].median(axis=1)
                else:
                    missing_features.append(feature)
            else:
                missing_features.append(feature)
    
    if missing_features:
        st.error(f"The following model features are missing or improperly mapped: {missing_features}")
        st.stop()
    
    # Create a new DataFrame with mapped and aggregated features
    mapped_df = pd.DataFrame(mapped_columns)
    return mapped_df
