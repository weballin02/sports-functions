# app.py

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
import streamlit as st
import json
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import plotly.express as px
import pickle
import os
import sqlite3
from utils.user_database import initialize_database, add_user, verify_user, get_user_id, save_model, load_models, load_model_by_name
from utils.auth import hash_password, check_password
from utils.column_mapper import interactive_column_mapping, apply_column_mapping

# Ensure the directory exists before initializing the database
data_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define the path to the database
DB_PATH = os.path.join(data_dir, 'users.db')

# Call the function to initialize the database
initialize_database()

# Define the path to the SQLite database (for direct access in login_page)
DB_PATH = os.path.join(os.getcwd(), 'data', 'users.db')

def registration_page():
    st.header("Create an Account")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    password_confirm = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if not username or not password or not password_confirm:
            st.error("Please fill out all fields.")
        elif password != password_confirm:
            st.error("Passwords do not match.")
        else:
            hashed_pw = hash_password(password)
            success = add_user(username, hashed_pw)
            if success:
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Please choose a different one.")

def login_page():
    st.header("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            # Retrieve hashed password from the database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            conn.close()
            
            if result and check_password(password, result[0]):
                # Successful login
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

def fetch_seasonal_data(sport, season_years):
    """
    Dummy function to mimic data fetching.
    Replace this with your actual data fetching logic.
    """
    # For demonstration, create a dummy DataFrame
    if sport == "NFL":
        data = {
            'completions': np.random.randint(0, 40, size=100),
            'attempts': np.random.randint(20, 60, size=100),
            'passing_yards': np.random.randint(100, 500, size=100),
            'passing_tds': np.random.randint(0, 10, size=100),
            'interceptions': np.random.randint(0, 5, size=100),
            # ... [add other NFL columns]
        }
    elif sport == "NBA":
        data = {
            'points': np.random.randint(0, 50, size=100),
            'rebounds': np.random.randint(0, 20, size=100),
            'assists': np.random.randint(0, 15, size=100),
            'steals': np.random.randint(0, 5, size=100),
            'blocks': np.random.randint(0, 5, size=100),
            # ... [add other NBA columns]
        }
    else:
        data = {}
    
    return pd.DataFrame(data)

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

def monte_carlo_simulation_page():
    st.header("Monte Carlo Simulation")
    st.write("""
    Perform Monte Carlo simulations on sports player statistics for a selected season.
    Select the sport, statistics, adjust weights, aggregate features if desired, and run simulations to estimate points scored.
    """)
    
    # Sidebar for user inputs
    st.sidebar.header("Simulation Settings")
    
    # Sport Selector
    sport = st.sidebar.selectbox("Select Sport", options=["NFL", "NBA"])
    
    # Select Season Year(s)
    season_year = st.sidebar.number_input("Select Season Year", min_value=2000, max_value=2100, value=2023)
    
    # Option to fetch data using the selected sport and season
    fetch_data_option = st.sidebar.checkbox("Fetch Data Using Selected Sport", value=True)
    
    # Fetch seasonal data
    if fetch_data_option:
        seasonal_data = fetch_seasonal_data(sport, [season_year])
        if seasonal_data.empty:
            st.warning("No data available for the selected sport and season.")
            st.stop()
        else:
            st.write(f"Fetched {sport} Data for Season {season_year}:")
            st.write(seasonal_data.head())
    else:
        # Option to upload data manually
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                seasonal_data = pd.read_csv(uploaded_file)
            else:
                seasonal_data = pd.read_excel(uploaded_file)
            st.write("Data Preview:")
            st.write(seasonal_data.head())
        else:
            st.warning("Please either fetch data using the selected sport or upload a dataset.")
            st.stop()
    
    # Show Available Columns Button
    if st.sidebar.button("Show Available Columns"):
        st.subheader(f"Available Columns for {sport} Season {season_year}")
        columns = seasonal_data.columns.tolist()
        columns_df = pd.DataFrame(columns, columns=["Column Names"])
        st.dataframe(columns_df)
    
    # Define available statistics per sport
    available_stats_dict = {
        'NFL': [
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'interceptions',
            'sacks',
            'sack_yards',
            'sack_fumbles',
            'sack_fumbles_lost',
            'passing_air_yards',
            'passing_yards_after_catch',
            'passing_first_downs',
            'passing_epa',
            'passing_2pt_conversions',
            'pacr',
            'dakota',
            'carries',
            'rushing_yards',
            'rushing_tds',
            'rushing_fumbles',
            'rushing_fumbles_lost',
            'rushing_first_downs',
            'rushing_epa',
            'rushing_2pt_conversions',
            'receptions',
            'targets',
            'receiving_yards',
            'receiving_tds',
            'receiving_fumbles',
            'receiving_fumbles_lost',
            'receiving_air_yards',
            'receiving_yards_after_catch',
            'receiving_first_downs',
            'receiving_epa',
            'receiving_2pt_conversions',
            'racr',
            'target_share',
            'air_yards_share',
            'wopr_x',
            'special_teams_tds',
            'fantasy_points',
            'fantasy_points_ppr',
            'games',
            'tgt_sh',
            'ay_sh',
            'yac_sh',
            'wopr_y',
            'ry_sh',
            'rtd_sh',
            'rfd_sh',
            'rtdfd_sh',
            'dom',
            'w8dom',
            'yptmpa',
            'ppr_sh'
        ],
        'NBA': [
            'points',
            'rebounds',
            'assists',
            'steals',
            'blocks',
            'turnovers',
            'field_goals_made',
            'field_goals_attempted',
            'free_throws_made',
            'free_throws_attempted',
            'three_pointers_made',
            'three_pointers_attempted',
            'minutes_played',
            'personal_fouls',
            # Add more NBA-specific statistics as needed
        ]
        # Add more sports as needed
    }
    
    available_stats = available_stats_dict.get(sport, [])
    
    # Dynamic features selection based on saved model or manual selection
    # Option to select a saved model
    st.sidebar.subheader("Use a Saved Model for Weights")
    saved_models = load_models(get_user_id(st.session_state['username']))
    # Filter models based on sport to avoid irrelevant models
    model_names = [model[0] for model in saved_models if f"Sport: {sport}" in model[0]]
    
    selected_model = st.sidebar.selectbox("Select a Saved Model", options=["None"] + model_names)
    
    if selected_model != "None":
        # Load the selected model
        model_data_str = load_model_by_name(get_user_id(st.session_state['username']), selected_model)
        if model_data_str:
            model_data = json.loads(model_data_str)
            mapping = model_data['mapping']
            weight_adjustments = model_data['weights']
            st.success(f"Model '{selected_model}' loaded successfully!")
        else:
            st.error(f"Failed to load model '{selected_model}'.")
            st.stop()
    else:
        # Manual selection of features and weights
        selected_features = st.sidebar.multiselect(
            "Select Statistics for Simulation",
            options=[stat for stat in available_stats if stat in seasonal_data.columns],
            default=[available_stats[0], available_stats[1]] if len(available_stats) >=2 else available_stats
        )

        if not selected_features:
            st.sidebar.warning("Please select at least one statistic to proceed.")
            st.stop()

        # Interactive Column Mapping with Aggregation
        mapping = interactive_column_mapping(sport, selected_features, seasonal_data)

        # Check if all features are mapped
        incomplete_mappings = [feat for feat, details in mapping.items()
                               if (details['type'] == 'single' and not details['column']) or
                               (details['type'] == 'aggregation' and (not details['columns'] or not details['method'] or not details['aggregated_feature_name']))]
        if incomplete_mappings:
            st.sidebar.warning("Please complete all mappings before proceeding.")
            st.stop()

        # Apply the column mapping
        mapped_simulated_stats = apply_column_mapping(mapping, seasonal_data)

        # Let user select which features to include
        selected_features = st.sidebar.multiselect(
            "Select Features to Include in Simulation",
            options=mapping.keys(),
            default=mapping.keys()  # All features by default
        )

        # Allow user to adjust weights
        st.sidebar.header("Points Calculation Weights")
        weight_adjustments = {}
        for feature in selected_features:
            default_weight = 1.0  # Default weight
            weight = st.sidebar.number_input(
                f"Weight for {feature.replace('_', ' ').title()}",
                value=default_weight,
                format="%.2f",
                key=f"weight_{feature}"
            )
            weight_adjustments[feature] = weight

    # ----------------------------
    # Model Management Section
    # ----------------------------
    st.sidebar.header("Model Management")
    
    # Save Current Model
    st.sidebar.subheader("Save Current Model")
    model_name = st.sidebar.text_input("Model Name")
    
    # Serialize the model data
    model_data = {
        'mapping': mapping,
        'weights': weight_adjustments
    }
    model_json = json.dumps(model_data)
    
    if st.sidebar.button("Save Model"):
        if not model_name:
            st.sidebar.error("Please provide a name for the model.")
        else:
            user_id = get_user_id(st.session_state['username'])
            success = save_model(user_id, model_name, model_json)
            if success:
                st.sidebar.success(f"Model '{model_name}' saved successfully!")
            else:
                st.sidebar.error(f"Failed to save model '{model_name}'. It might already exist.")
    
    # Load Existing Models
    st.sidebar.subheader("Load a Saved Model")
    saved_models = load_models(get_user_id(st.session_state['username']))
    
    if saved_models:
        model_names = [model[0] for model in saved_models]
        selected_load_model = st.sidebar.selectbox("Select a Model to Load", options=model_names)
        
        if st.sidebar.button("Load Model"):
            loaded_model_data_str = load_model_by_name(get_user_id(st.session_state['username']), selected_load_model)
            if loaded_model_data_str:
                loaded_model_data = json.loads(loaded_model_data_str)
                mapping = loaded_model_data['mapping']
                weight_adjustments = loaded_model_data['weights']
                st.success(f"Model '{selected_load_model}' loaded successfully!")
            else:
                st.error(f"Failed to load model '{selected_load_model}'.")
    else:
        st.sidebar.info("No saved models found.")
    
    # ----------------------------
    # Number of Simulations
    # ----------------------------
    # Number of simulations using slider (10,000 to 1,000,000)
    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=10_000,
        max_value=1_000_000,
        value=100_000,
        step=10_000
    )
    
    # ----------------------------
    # Run Simulation Button
    # ----------------------------
    if st.sidebar.button("Run Monte Carlo Simulation"):
        if not selected_features:
            st.warning("Please select at least one feature to run the simulation.")
        else:
            with st.spinner("Running simulations..."):
                # Apply the column mapping if not loaded from a saved model
                if selected_model == "None":
                    # mapped_simulated_stats already defined
                    pass
                else:
                    # When a model is loaded, mapped_simulated_stats was defined earlier
                    pass
                
                # Calculate stat ranges based on selected features
                stats_summary = calculate_stat_ranges(mapped_simulated_stats, selected_features)

                # Perform simulations
                simulated_stats_df = simulate_stats(stats_summary, num_simulations)

                # Calculate points
                points = calculate_points(simulated_stats_df[selected_features], weight_adjustments)

                # Summary statistics
                summary = {
                    'Mean Points': points.mean(),
                    'Median Points': points.median(),
                    'Standard Deviation': points.std(),
                    'Minimum Points': points.min(),
                    'Maximum Points': points.max(),
                    '5th Percentile': points.quantile(0.05),
                    '25th Percentile': points.quantile(0.25),
                    '75th Percentile': points.quantile(0.75),
                    '95th Percentile': points.quantile(0.95)
                }

                summary_df_sim = pd.Series(summary).to_frame(name="Value")
                st.subheader("Simulation Summary")
                st.table(summary_df_sim.style.format({"Value": "{:.2f}"}))

                # Plotting the distribution
                st.subheader("Points Distribution")
                fig = px.histogram(points, nbins=50, title="Histogram of Simulated Points")
                fig.update_layout(xaxis_title="Points", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)

                # Box plot
                st.subheader("Box Plot of Simulated Points")
                fig_box = px.box(y=points, title="Box Plot of Simulated Points")
                fig_box.update_layout(yaxis_title="Points")
                st.plotly_chart(fig_box, use_container_width=True)

                # Scatter Plots Section
                st.subheader("Scatter Plots for Selected Statistics")

                # Allow user to select two stats for scatter plot
                if len(selected_features) >= 2:
                    scatter_options = [(x, y) for idx, x in enumerate(selected_features) for y in selected_features[idx + 1:]]
                    scatter_stats = st.selectbox(
                        "Select two statistics for scatter plot",
                        options=scatter_options,
                        format_func=lambda x: f"{x[0].replace('_', ' ').title()} vs {x[1].replace('_', ' ').title()}"
                    )

                    if scatter_stats:
                        x_stat, y_stat = scatter_stats
                        # Simulated data scatter plot
                        fig_scatter = generate_scatter_plot(simulated_stats_df, x_stat, y_stat)
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Pairwise Scatter Matrix
                st.subheader("Pairwise Scatter Matrix")
                pairwise_fig = generate_pairwise_scatter_matrix(simulated_stats_df, selected_features)
                st.plotly_chart(pairwise_fig, use_container_width=True)

                # Downloadable Reports Section
                st.subheader("Download Simulation Reports")

                # Prepare simulation results DataFrame
                simulation_results_df = pd.DataFrame({
                    'Simulation_Number': range(1, num_simulations + 1),
                    'Points': points
                })

                # Convert DataFrames to CSV
                simulation_csv = convert_df_to_csv(simulation_results_df)
                summary_csv = convert_df_to_csv(summary_df_sim.reset_index().rename(columns={'index': 'Statistic'}))

                # Download buttons
                st.download_button(
                    label="Download Simulation Results (CSV)",
                    data=simulation_csv,
                    file_name='simulation_results.csv',
                    mime='text/csv',
                )

                st.download_button(
                    label="Download Summary Statistics (CSV)",
                    data=summary_csv,
                    file_name='simulation_summary.csv',
                    mime='text/csv',
                )

def calculate_stat_ranges(df, selected_stats):
    """
    Calculates min, max, mean, and standard deviation for each selected stat.
    """
    stats_summary = {}
    for stat in selected_stats:
        if stat in df.columns:
            stats_summary[stat] = {
                'min': df[stat].min(),
                'max': df[stat].max(),
                'mean': df[stat].mean(),
                'std': df[stat].std()
            }
        else:
            st.warning(f"Statistic '{stat}' not found in the data.")
    return stats_summary

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=1):
    """
    Generates random numbers from a truncated normal distribution.
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size)

def simulate_stats(stats_summary, num_simulations):
    """
    Performs Monte Carlo simulations with bounded randomization.
    """
    simulation_data = {}
    for stat, summary in stats_summary.items():
        mean = summary['mean']
        std = summary['std']
        min_val = summary['min']
        max_val = summary['max']
        # Handle cases where std is zero to avoid division by zero
        if std == 0:
            simulated = np.full(num_simulations, mean)
        else:
            simulated = get_truncated_normal(mean, std, min_val, max_val, num_simulations)
        simulation_data[stat] = simulated
    return pd.DataFrame(simulation_data)

def calculate_points(simulated_df, weights):
    """
    Calculates points scored based on a weighted formula.
    """
    # Ensure that weights align with the columns in simulated_df
    relevant_weights = {stat: weights.get(stat, 0) for stat in simulated_df.columns}
    weights_series = pd.Series(relevant_weights)
    points = simulated_df.mul(weights_series, axis=1).sum(axis=1)
    return points

def generate_scatter_plot(df, x_stat, y_stat):
    """
    Generates a scatter plot for two selected statistics.
    """
    fig = px.scatter(df, x=x_stat, y=y_stat, trendline="ols",
                     title=f"Scatter Plot: {x_stat.replace('_', ' ').title()} vs {y_stat.replace('_', ' ').title()}",
                     labels={
                         x_stat: x_stat.replace('_', ' ').title(),
                         y_stat: y_stat.replace('_', ' ').title()
                     })
    return fig

def generate_pairwise_scatter_matrix(df, selected_stats):
    """
    Generates a pairwise scatter matrix for the selected statistics.
    """
    fig = px.scatter_matrix(df,
                            dimensions=selected_stats,
                            title="Pairwise Scatter Matrix",
                            labels={stat: stat.replace('_', ' ').title() for stat in selected_stats})
    return fig

def convert_df_to_csv(df):
    """
    Converts a DataFrame to a CSV format.
    """
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("Sports Monte Carlo Simulation App")
    
    # Sidebar Navigation
    menu = ["Login", "Register", "Monte Carlo Simulation", "Correlation Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if choice == "Register":
        registration_page()
    elif choice == "Login":
        login_page()
    elif choice == "Monte Carlo Simulation":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            # Call your Monte Carlo Simulation page function
            monte_carlo_simulation_page()
            
            # Option to log out
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access the Monte Carlo Simulation.")
    elif choice == "Correlation Analysis":
        if st.session_state['logged_in']:
            st.sidebar.write(f"Logged in as: {st.session_state['username']}")
            # Call your Correlation Analysis page function
            correlation_analysis_page()
            
            # Option to log out
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.success("Logged out successfully!")
        else:
            st.warning("Please log in to access the Correlation Analysis.")

if __name__ == "__main__":
    main()
