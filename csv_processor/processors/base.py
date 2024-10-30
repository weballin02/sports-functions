# csv_processor/processors/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, config):
        self.config = config
        self.team_mappings = self._load_team_mappings()
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe according to specific rules"""
        pass
    
    def standardize_team_names(self, df: pd.DataFrame, team_cols: list) -> pd.DataFrame:
        """Standardize team names across all processors"""
        for col in team_cols:
            if col in df.columns:
                df[col] = df[col].map(self.team_mappings).fillna(df[col])
        return df
    
    def _load_team_mappings(self):
        """Load team name mappings from configuration"""
        return self.config.get("team_mappings", {})