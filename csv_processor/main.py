import os
import signal
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List
import logging
from .utils.helpers import load_config, get_csv_files, save_dataframe
from .processors.base import BaseProcessor

class DataProcessor:
    def __init__(self, config_path: str, input_dir: str, output_dir: str, batch_size: int = 1000, 
                 master_save_interval: timedelta = timedelta(hours=1), periodic_save_interval: timedelta = timedelta(minutes=10)):
        self.config = load_config(config_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.master_save_interval = master_save_interval
        self.periodic_save_interval = periodic_save_interval
        self.batch_counter = 0
        self.expanded_data: List[pd.DataFrame] = []
        self.last_master_save_time = datetime.now()
        self.last_periodic_save_time = datetime.now()
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

        self.processors = self._initialize_processors()
        self._setup_logging()
        self._setup_signal_handlers()

    def _initialize_processors(self) -> dict:
        from .processors import NFLProcessor, NBAProcessor
        return {
            'nfl': NFLProcessor(self.config),
            'nba': NBAProcessor(self.config)
        }

    def _setup_logging(self):
        """Configure logging"""
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists for logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'processor.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._cleanup_handler)
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        self.logger.info("Shutdown signal received. Cleaning up...")
        self.save_remaining_data()
        self.append_to_master_file()
        self.logger.info("Cleanup complete. Exiting.")
        exit(0)

    def process_file(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            file_type = self._identify_file_type(file_path)
            if not file_type:
                self.logger.warning(f"Unknown file type: {file_path}")
                return None
                
            processor = self.processors.get(file_type)
            if not processor:
                self.logger.warning(f"No processor found for type: {file_type}")
                return None
            
            df = pd.read_csv(
                file_path,
                dtype=self.config[file_type].get('dtypes'),
                parse_dates=self.config[file_type].get('parse_dates')
            )
            
            processed_df = processor.process(df)
            self.logger.info(f"Successfully processed {file_path}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return None

    def _identify_file_type(self, file_path: str) -> Optional[str]:
        filename = os.path.basename(file_path)
        for file_type, config in self.config.items():
            if any(pattern in filename for pattern in config.get('file_patterns', [])):
                return file_type
        return None

    def save_batch(self):
        if not self.expanded_data:
            return
        try:
            df = pd.concat(self.expanded_data, ignore_index=True)
            batch_path = os.path.join(self.output_dir, f'batch_{self.batch_counter}.parquet')
            save_dataframe(df, batch_path)
            self.logger.info(f"Saved batch {self.batch_counter}")
            self.expanded_data = []
            self.batch_counter += 1
        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}", exc_info=True)

    def append_to_master_file(self):
        try:
            batch_files = get_csv_files(self.output_dir, "batch_*.parquet")
            if not batch_files:
                return
                
            batch_dfs = [pd.read_parquet(f) for f in batch_files]
            combined_df = pd.concat(batch_dfs, ignore_index=True)
            master_path = os.path.join(self.output_dir, 'master.parquet')
            if os.path.exists(master_path):
                master_df = pd.read_parquet(master_path)
                combined_df = pd.concat([master_df, combined_df], ignore_index=True)
                combined_df.drop_duplicates(inplace=True)
            save_dataframe(combined_df, master_path)
            self.logger.info("Successfully updated master file")
            for file in batch_files:
                os.remove(file)
            self.batch_counter = 0
        except Exception as e:
            self.logger.error(f"Error updating master file: {str(e)}", exc_info=True)

    def save_remaining_data(self):
        if self.expanded_data:
            self.save_batch()
    
    def process_files(self):
        try:
            csv_files = get_csv_files(self.input_dir)
            for file_path in csv_files:
                processed_df = self.process_file(file_path)
                if processed_df is not None:
                    self.expanded_data.append(processed_df)
                    if len(self.expanded_data) >= self.batch_size:
                        self.save_batch()
                current_time = datetime.now()
                if current_time - self.last_master_save_time >= self.master_save_interval:
                    self.append_to_master_file()
                    self.last_master_save_time = current_time
                if current_time - self.last_periodic_save_time >= self.periodic_save_interval:
                    self.save_batch()
                    self.last_periodic_save_time = current_time
            self.save_remaining_data()
            self.append_to_master_file()
        except Exception as e:
            self.logger.error(f"Error in processing loop: {str(e)}", exc_info=True)
            raise

def run_processor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "config.json")  # Corrected path
    input_dir = os.path.join(base_dir, "data", "input")
    output_dir = os.path.join(base_dir, "data", "output")

    processor = DataProcessor(
        config_path=config_path,
        input_dir=input_dir,
        output_dir=output_dir
    )
    print("Starting file processing...")
    processor.process_files()
    print("File processing completed.")



if __name__ == "__main__":
    run_processor()

