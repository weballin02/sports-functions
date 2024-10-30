# csv_processor/processors/nfl.py
from .base import BaseProcessor
import pandas as pd

class NFLProcessor(BaseProcessor):
    """Processor for NFL game data"""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize team names
        df = self.standardize_team_names(df, ['home_team', 'away_team'])
        
        # Calculate point differential
        if all(col in df.columns for col in ['home_score', 'away_score']):
            df['point_diff'] = df['home_score'] - df['away_score']
            
        # Add win/loss/tie column
        df['result'] = 'tie'
        df.loc[df['point_diff'] > 0, 'result'] = 'win'
        df.loc[df['point_diff'] < 0, 'result'] = 'loss'
        
        return df