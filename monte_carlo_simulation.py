# monte_carlo_simulation.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import plotly.express as px
import pickle
from utils.database import get_saved_models, load_model
from utils.sports_data import fetch_sport_data
from utils.column_mapper import interactive_column_mapping, apply_column_mapping


# ============================
# Helper Functions
# ============================

@st.cache_data
def fetch_seasonal_data(sport, season_years):
    """
    Fetches seasonal data for the given sport and season years.
    """
    try:
        seasonal_data = fetch_sport_data(sport, season_years)
        return seasonal_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

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

# ============================
# Streamlit App Layout
# ============================

def monte_carlo_simulation_page():
    st.title("Monte Carlo Simulation")
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
    saved_models = get_saved_models(0)  # Using user_id 0 as placeholder
    # Filter models based on sport to avoid irrelevant models
    model_names = [model_name for model_name, meta in saved_models if f"Sport: {sport}" in meta]
    
    selected_model = st.sidebar.selectbox("Select a Saved Model", options=["None"] + model_names)
    
    if selected_model != "None":
        # Load the selected model
        loaded_model, loaded_scaler = load_model(0, selected_model)
        if loaded_model and loaded_scaler:
            # Deserialize the feature names and weights
            if hasattr(loaded_model, 'feature_names_in_'):
                model_features = loaded_model.feature_names_in_
                # Ensure feature_names is a list
                if isinstance(model_features, (list, tuple)):
                    pass  # Already a list or tuple
                elif hasattr(model_features, 'tolist'):
                    model_features = model_features.tolist()
                else:
                    st.error("Unsupported type for feature names.")
                    model_features = []
            else:
                st.error("Loaded model does not have 'feature_names_in_' attribute.")
                model_features = []
            
            weights = loaded_model.coef_

            # Create a DataFrame for weights
            weights_df = pd.DataFrame({
                'Feature': model_features,
                'Weight': weights
            })

            # Display the loaded weights
            st.sidebar.subheader("Model Feature Weights")
            st.sidebar.write(weights_df)

            # Interactive Column Mapping with Aggregation
            mapping = interactive_column_mapping(sport, model_features, seasonal_data)

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
                options=model_features,
                default=model_features  # All features by default
            )

            # Prepopulate weights based on the selected model
            weight_adjustments = {}
            for feature in selected_features:
                if feature in weights_df['Feature'].values:
                    weight = weights_df.loc[weights_df['Feature'] == feature, 'Weight'].values[0]
                    weight_adjustments[feature] = weight
                else:
                    weight_adjustments[feature] = 0  # Default weight if feature not found
        else:
            st.sidebar.error("Failed to load the selected model.")
            selected_features = []
            weight_adjustments = {}
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

        # Define or fetch weights based on correlation analysis
        # For demonstration, use a predefined weight dictionary.
        # Replace this with your correlation-based weights as needed.
        default_weights_dict = {
            'NFL': {
                'completions': 0.5,
                'attempts': 0.2,
                'passing_yards': 0.04,
                'passing_tds': 6.0,
                'interceptions': -2.0,
                # ... [other NFL weights]
            },
            'NBA': {
                'points': 1.0,
                'rebounds': 1.2,
                'assists': 1.5,
                'steals': 2.0,
                'blocks': 2.5,
                'turnovers': -1.5,
                # ... [other NBA weights]
            }
            # Add more sports and their default weights as needed
        }

        default_weights = default_weights_dict.get(sport, {})

        # Allow user to adjust weights if desired
        st.sidebar.header("Points Calculation Weights")
        weight_adjustments = {}
        for stat in selected_features:
            default_weight = default_weights.get(stat, 0)
            weight = st.sidebar.number_input(
                f"Weight for {stat.replace('_', ' ').title()}",
                value=default_weight,
                format="%.2f",
                key=f"weight_{stat}"
            )
            weight_adjustments[stat] = weight

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
                        # Original data scatter plot
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

    # Option to view and load saved models is now correctly inside the function
    # This code should be indented to be inside the monte_carlo_simulation_page function
    # To avoid confusion, ensure all such code is inside the function
