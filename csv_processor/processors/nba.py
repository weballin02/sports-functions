# csv_processor/processors/nba.py
from .base import BaseProcessor
import pandas as pd

class NBAProcessor(BaseProcessor):
    """Processor for NBA player statistics"""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize team names
        df = self.standardize_team_names(df, ['team'])
        
        # Calculate efficiency rating if required columns exist
        required_cols = ['points', 'rebounds', 'assists', 'minutes_played']
        if all(col in df.columns for col in required_cols):
            df['efficiency'] = (
                (df['points'] + df['rebounds'] + df['assists']) / 
                df['minutes_played'].clip(lower=1)
            )
        
        return df