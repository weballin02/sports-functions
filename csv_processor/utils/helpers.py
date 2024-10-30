# csv_processor/utils/helpers.py
import os
import glob
import json
from datetime import datetime
import pandas as pd

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_csv_files(directory: str, pattern: str = "*.csv") -> list:
    """Get list of CSV files matching pattern in directory"""
    return glob.glob(os.path.join(directory, pattern))

def save_dataframe(df: pd.DataFrame, path: str, format: str = "parquet") -> None:
    """Save DataFrame to specified format"""
    if format == "parquet":
        df.to_parquet(path, engine='fastparquet', compression='gzip', index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")