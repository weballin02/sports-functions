from nba_api.stats.static import teams
import pandas as pd

def get_all_teams():
    # Fetch all NBA teams
    nba_teams = teams.get_teams()
    
    # Convert to DataFrame for easy viewing and manipulation
    teams_df = pd.DataFrame(nba_teams)
    
    # Display team ID, full name, and abbreviation
    return teams_df[['id', 'full_name', 'abbreviation']]

# Run the function and display the teams DataFrame
if __name__ == "__main__":
    teams_df = get_all_teams()
    print(teams_df)

