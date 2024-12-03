import numpy as np
import pandas as pd
import os
import sys

from dataclasses import dataclass

from src.logger import get_logger
from src.exception import CustomException

from sklearn.model_selection import train_test_split

logger = get_logger('data ingestion')

@dataclass
class DataIngestionConfig:
    raw_path: str = os.path.join('artifacts', 'data.csv')
    train_path: str = os.path.join('artifacts', 'train.csv')
    test_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def load_data(self):
        '''
        It loads the data from directory and split it and then save it in artifacts folder

        Returns:
            train_path, test_path
        '''
        try:
            logger.info('Data Ingestion started')
            FILE_PATH = 'Notebooks/data/cleaned_data.csv'

            logger.info('Loading raw data')
            df = pd.read_csv(FILE_PATH)

            logger.info('Splitting data into training and testing part')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path), exist_ok=True)

            logger.info('Saving raw data to artifacts')
            df.to_csv(self.data_ingestion_config.raw_path, index=False)

            logger.info('Saving train data to artifacts')
            train_data.to_csv(self.data_ingestion_config.train_path, index=False)

            logger.info('Saving test data to artifacts')
            test_data.to_csv(self.data_ingestion_config.test_path, index=False)

            logger.info('Data Ingestion Completed')

            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)