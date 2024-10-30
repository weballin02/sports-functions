# csv_processor/__init__.py
from .main import DataProcessor
from .processors import NFLProcessor, NBAProcessor

__version__ = "0.1.0"
__all__ = ["DataProcessor", "NFLProcessor", "NBAProcessor"]
