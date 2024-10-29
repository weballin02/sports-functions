# correlation_analysis.py

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import BytesIO
from utils.correlation_utils import (
    calculate_correlation,
    calculate_vif,
    check_numeric_columns
)
import pickle
from utils.database import save_model, get_saved_models, load_model
from utils.sports_data import fetch_sport_data

def aggregate_play_by_play_data(df, group_columns, agg_columns):
    """
    Aggregates play-by-play data based on selected group and aggregate columns.
    """
    agg_dict = {col: 'sum' for col in agg_columns}
    aggregated_df = df.groupby(group_columns).agg(agg_dict).reset_index()
    num_groups = aggregated_df.shape[0]
    return aggregated_df, num_groups

def calculate_feature_importance(df, features, target_column):
    """
    Calculates feature importance using Ridge Regression and VIF.
    """
    df_clean = df.dropna(subset=features + [target_column])
    X = df_clean[features]
    y = df_clean[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = calculate_vif(pd.DataFrame(X_scaled, columns=features))
    st.write("VIF (Variance Inflation Factor) to check multicollinearity:")
    st.write(vif_data)
    
    model = Ridge()
    model.fit(X_scaled, y)
    
    feature_importance = model.coef_
    model.feature_names_in_ = features  # Ensure this is a list
    
    return feature_importance, scaler, model

def scale_formula_per_game(feature_importance, num_games):
    """
    Scales feature importance based on the number of games.
    """
    return feature_importance / num_games

def generate_excel(importance_df, formula_str, features):
    """
    Generates an Excel file with feature importance and the prediction formula.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        df_formula = pd.DataFrame({'Prediction Formula': [formula_str]})
        df_formula.to_excel(writer, sheet_name='Formula', index=False)
    processed_data = output.getvalue()
    return processed_data

def correlation_analysis_page():
    # Streamlit App
    st.title("Correlation and Feature Importance Analysis")
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Settings")
    
    # Sport Selector
    sport = st.sidebar.selectbox("Select Sport", options=["NFL", "NBA"])
    
    # Select Season Year(s)
    season_year = st.sidebar.number_input("Select Season Year", min_value=2000, max_value=2100, value=2023)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())
    
    # Option to fetch data using the selected sport and season
    fetch_data_option = st.sidebar.checkbox("Fetch Data Using Selected Sport", value=False)
    
    if fetch_data_option:
        try:
            df = fetch_sport_data(sport, [season_year])
            st.write(f"Fetched and Cleaned {sport} Data for Season {season_year}:")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
    
    if 'df' not in locals():
        st.warning("Please upload a dataset or fetch data using the selected sport.")
        st.stop()
    
    aggregate_option = st.checkbox("Aggregate Data (e.g., for Play-by-Play)", value=False)
    num_games = 1  # Default to 1 if no aggregation is applied
    
    if aggregate_option:
        group_columns = st.multiselect("Select columns to group by (e.g., game_id, team)", df.columns)
        agg_columns = st.multiselect("Select columns to aggregate (e.g., rushing_yards, passing_yards)", df.select_dtypes(include='number').columns)
    
        if group_columns and agg_columns:
            df, num_games = aggregate_play_by_play_data(df, group_columns, agg_columns)
            st.write(f"Aggregated Data Preview (Number of unique groups: {num_games}):")
            st.write(df.head())
    
    st.write("Proceeding with correlation and feature importance analysis...")
    target_columns = st.multiselect("Select Target Column (e.g., fantasy_points)", df.select_dtypes(include='number').columns)
    
    if target_columns:
        feature_columns = st.multiselect(
            "Select Feature Columns (e.g., passing_yards, rushing_yards)",
            [col for col in df.select_dtypes(include='number').columns if col not in target_columns]
        )
    
        if feature_columns:
            correlation_result = calculate_correlation(df[feature_columns + target_columns], target_columns)
            st.write("Correlation Analysis Result:")
            st.write(correlation_result)
    
            st.write("Correlation Heatmap:")
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_result, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
    
            target_column = st.selectbox("Select a single Target Column for Feature Importance", target_columns)
    
            if target_column:
                final_feature_columns = [col for col in feature_columns if col != target_column]
                feature_importance, scaler, model = calculate_feature_importance(df, final_feature_columns, target_column)
    
                if aggregate_option:
                    feature_importance = scale_formula_per_game(feature_importance, num_games)
    
                st.write(f"Feature Importance for predicting {target_column}:")
                importance_df = pd.DataFrame({
                    'Feature': final_feature_columns,
                    'Importance (Weight)': feature_importance
                }).sort_values(by='Importance (Weight)', ascending=False)
    
                st.write(importance_df)
    
                st.write("Weighted Formula for Prediction:")
                formula_str = f"{target_column} = "
                for i, feature in enumerate(importance_df['Feature']):
                    weight = importance_df['Importance (Weight)'].iloc[i]
                    if i > 0:
                        formula_str += " + "
                    formula_str += f"({weight:.3f}) * {feature}"
                st.write(formula_str)
    
                # Option to save model
                save_model_option = st.checkbox("Save trained model?", value=False)
                if save_model_option:
                    model_name = st.text_input("Enter a name for your model:")
                    if st.button("Save Model"):
                        if model_name:
                            try:
                                model_data = pickle.dumps((model, scaler))
                                metadata = f"Sport: {sport}, Season: {season_year}, Model trained on {target_column} with features {final_feature_columns}"
                                save_model(0, model_name, model_data, metadata)  # Using user_id 0 as placeholder
                                st.success("Model saved successfully!")
                            except Exception as e:
                                st.error(f"Error saving model: {e}")
                        else:
                            st.warning("Please enter a valid model name.")
    
                # Generate Excel file with results and formula
                excel_data = generate_excel(importance_df, formula_str, final_feature_columns)
                st.download_button(
                    label="Download Results as Excel", 
                    data=excel_data, 
                    file_name="feature_importance_analysis.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Option to view and load saved models
    st.write("## Your Saved Models")
    saved_models = get_saved_models(0)  # Using user_id 0 as placeholder
    
    if saved_models:
        for idx, (model_name, model_metadata) in enumerate(saved_models):
            st.write(f"### Model: {model_name}")
            st.write(f"**Metadata:** {model_metadata}")
            if st.button(f"Load {model_name}", key=f"load_model_{idx}"):
                loaded_model, loaded_scaler = load_model(0, model_name)
                if loaded_model and loaded_scaler:
                    st.success(f"Loaded model: {model_name}")
                    st.write(f"Metadata: {model_metadata}")
                    
                    # Prompt user to enter new data points for prediction
                    st.write(f"#### Predict using {model_name}")
                    new_data_input = st.text_area(
                        f"Input new data for prediction (comma-separated, matching features):", 
                        key=f"new_data_{idx}"
                    )
    
                    if new_data_input:
                        try:
                            new_data = [float(x.strip()) for x in new_data_input.split(',')]
                            new_data_scaled = loaded_scaler.transform([new_data])
                            prediction = loaded_model.predict(new_data_scaled)
                            st.write(f"**Prediction for the provided data:** {prediction[0]:.2f}")
                        except Exception as e:
                            st.error(f"Error in prediction: {e}")
                else:
                    st.error("Failed to load the model. Please try again.")
    else:
        st.write("No models saved yet.")
