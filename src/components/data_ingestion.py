import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', 'raw', f'top_movies.csv-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    training_data_path: str = os.path.join('data', 'train', f'training_data.csv-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    test_data_path: str = os.path.join('data', 'test', f'test_data.csv-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv('externals/top_movies.csv')
            logging.info("Raw data file read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.training_data_path, index=False, header=True)
            logging.info(f"Training data saved to {self.ingestion_config.training_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            logging.info("Data Ingestion completed successfully")
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.training_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()