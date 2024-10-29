import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2, LeagueDashTeamStats
from datetime import datetime

# Define team mappings
team_names = {
    '1610612737': 'ATL', '1610612738': 'BOS', '1610612739': 'CLE',
    '1610612740': 'NOP', '1610612741': 'CHI', '1610612742': 'DAL',
    '1610612743': 'DEN', '1610612744': 'GSW', '1610612745': 'HOU',
    '1610612746': 'LAC', '1610612747': 'LAL', '1610612748': 'MIA',
    '1610612749': 'MIL', '1610612750': 'MIN', '1610612751': 'BKN',
    '1610612752': 'NYK', '1610612753': 'ORL', '1610612754': 'IND',
    '1610612755': 'PHI', '1610612756': 'PHX', '1610612757': 'POR',
    '1610612758': 'SAC', '1610612759': 'SAS', '1610612760': 'OKC',
    '1610612761': 'TOR', '1610612762': 'UTA', '1610612763': 'MEM',
    '1610612764': 'WAS', '1610612765': 'DET', '1610612766': 'CHA'
}

# Fetch team stats for predictions
def fetch_team_stats():
    stats = LeagueDashTeamStats().get_data_frames()[0]
    stats = stats[['TEAM_ID', 'TEAM_NAME', 'PTS', 'PLUS_MINUS', 'AST', 'TOV', 'STL']]
    stats['TEAM_ABBREVIATION'] = stats['TEAM_ID'].astype(str).map(team_names)
    stats.set_index('TEAM_ABBREVIATION', inplace=True)
    return stats

# Get today's NBA games and include provided time and broadcast info
def fetch_todays_games():
    today = datetime.now().strftime("%Y-%m-%d")
    games = ScoreboardV2(game_date=today).get_data_frames()[0]
    games = games[['GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'LIVE_PERIOD_TIME_BCAST', 
                   'NATL_TV_BROADCASTER_ABBREVIATION', 'HOME_TV_BROADCASTER_ABBREVIATION', 'AWAY_TV_BROADCASTER_ABBREVIATION']]
    
    # Map team IDs to abbreviations
    games['HOME_TEAM'] = games['HOME_TEAM_ID'].astype(str).map(team_names)
    games['AWAY_TEAM'] = games['VISITOR_TEAM_ID'].astype(str).map(team_names)
    
    # Format game date and time as is, without conversion
    games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST']).dt.strftime('%m/%d %I:%M %p')
    
    # Select relevant columns for display
    games_display = games[['GAME_DATE_EST', 'HOME_TEAM', 'AWAY_TEAM', 
                           'NATL_TV_BROADCASTER_ABBREVIATION', 'HOME_TV_BROADCASTER_ABBREVIATION', 'AWAY_TV_BROADCASTER_ABBREVIATION']]
    
    # Rename columns for clarity
    games_display.columns = ['Game Date', 'Home Team', 'Away Team', 'National TV', 'Home TV', 'Away TV']
    return games_display

# Prediction functions
def win_probability(home_stats, away_stats):
    home_score = home_stats['AST'] + home_stats['STL'] - home_stats['TOV']
    away_score = away_stats['AST'] + away_stats['STL'] - away_stats['TOV']
    return 1 / (1 + 10 ** ((away_score - home_score) / 400))

def predict_game_outcome(games, team_stats):
    predictions = []
    for _, game in games.iterrows():
        home_team = game['Home Team']
        away_team = game['Away Team']
        
        # Retrieve stats for home and away teams
        home_stats = team_stats.loc[home_team]
        away_stats = team_stats.loc[away_team]
        
        # Calculate win probability and expected margin
        home_win_prob = win_probability(home_stats, away_stats)
        expected_margin = (home_stats['PLUS_MINUS'] - away_stats['PLUS_MINUS']) / 2
        
        predictions.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(home_win_prob * 100, 2),
            "expected_margin_of_victory": round(expected_margin, 2)
        })
    
    return pd.DataFrame(predictions)

# Streamlit app layout
st.title("NBA Game Prediction App (Reverted to Original)")

# Fetch today's games and team stats
todays_games_df = fetch_todays_games()
team_stats_df = fetch_team_stats()

if not todays_games_df.empty and not team_stats_df.empty:
    st.subheader("Today's NBA Games")
    st.write(todays_games_df)  # Display games with provided game time and broadcast info

    # Predict outcomes
    predictions_df = predict_game_outcome(todays_games_df, team_stats_df)
    st.subheader("Predictions and Expected Outcomes")
    st.write(predictions_df)
else:
    st.write("No NBA games scheduled for today or unable to fetch team stats.")
