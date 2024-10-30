# process_data.py
from csv_processor.main import DataProcessor

processor = DataProcessor(
    config_path="config/config.json",
    input_dir="data/input",
    output_dir="data/output"
)
processor.process_files()

