import streamlit as st
import pandas as pd
from nfl_data_py import import_schedules, import_team_desc, import_weekly_data
from datetime import datetime

# Define team mappings using Team Description data
team_names = import_team_desc()[['team_abbr', 'team_id']].set_index('team_id')['team_abbr'].to_dict()

# Fetch cumulative team stats for predictions using available columns
def fetch_team_stats():
    weekly_data = import_weekly_data([2023])  # Adjust season as needed
    team_stats = weekly_data.groupby('recent_team').agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'fantasy_points': 'sum',
        'interceptions': 'sum'
    }).reset_index()
    
    team_stats['TEAM_ABBREVIATION'] = team_stats['recent_team'].map(team_names)
    team_stats.set_index('TEAM_ABBREVIATION', inplace=True)
    return team_stats

# Fetch upcoming NFL games correctly by filtering for future dates only
def fetch_upcoming_games():
    today = pd.to_datetime(datetime.now().date())
    schedules = import_schedules([2023])  # Adjust season as needed
    schedules['gameday'] = pd.to_datetime(schedules['gameday'], errors='coerce').dt.normalize()
    
    # Filter for future games only
    future_games = schedules[schedules['gameday'] > today].sort_values(by='gameday').head(5)
    
    if future_games.empty:
        print("No upcoming games found in the data for the selected season.")
        return pd.DataFrame()  # Return an empty DataFrame if no future games

    # Select relevant columns and format display
    games_display = future_games[['gameday', 'home_team', 'away_team', 'gametime', 'location', 'spread_line', 'over_odds', 'under_odds']]
    games_display['gameday'] = games_display['gameday'].dt.strftime('%m/%d %I:%M %p')
    
    # Rename columns for display
    games_display.columns = ['Game Date', 'Home Team', 'Away Team', 'Game Time', 'Location', 'Spread Line', 'Over Odds', 'Under Odds']
    return games_display

# Streamlit app layout
st.title("NFL Game Prediction App")

# Display upcoming games or error message
todays_or_upcoming_games_df = fetch_upcoming_games()
if not todays_or_upcoming_games_df.empty:
    st.subheader("Upcoming NFL Games")
    st.write(todays_or_upcoming_games_df)
else:
    st.write("No upcoming NFL games found in the schedule.")
