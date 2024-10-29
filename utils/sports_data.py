# utils/sports_data.py

import pandas as pd
import nfl_data_py as nfl
# Import other sport-specific libraries as needed, e.g., for NBA, MLB, etc.
# import nba_data_py as nba  # Example placeholder

def fetch_sport_data(sport, season_years):
    """
    Fetches seasonal data for the specified sport and season years.

    Parameters:
    - sport (str): The sport to fetch data for (e.g., 'NFL', 'NBA').
    - season_years (list): List of season years to fetch data for.

    Returns:
    - pd.DataFrame: The fetched and cleaned data.
    """
    if sport == 'NFL':
        try:
            seasonal_data = nfl.import_seasonal_data(season_years)
            if seasonal_data.empty:
                raise ValueError(f"No data found for NFL seasons: {season_years}")
            seasonal_data = nfl.clean_nfl_data(seasonal_data)
            return seasonal_data
        except Exception as e:
            raise RuntimeError(f"Error fetching NFL data: {e}")

    elif sport == 'NBA':
        # Placeholder for NBA data fetching logic
        # Implement similar to NFL using an appropriate library or API
        # Example:
        # seasonal_data = nba.import_seasonal_data(season_years)
        # seasonal_data = nba.clean_nba_data(seasonal_data)
        # return seasonal_data
        raise NotImplementedError("NBA data fetching is not yet implemented.")

    # Add more sports as needed
    else:
        raise ValueError(f"Unsupported sport: {sport}")
