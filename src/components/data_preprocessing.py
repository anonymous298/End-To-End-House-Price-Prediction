import os              
import sys
import numpy as np                  
import pandas as pd               

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import get_train_test_data, save_model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

logger = get_logger('data-preprocessing')

@dataclass
class DataPreprocessingConfig:
    preprocessor_path = os.path.join('model', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()

    def get_preprocessor_object(self, X):
        '''
        returns the preprocessor object from which we are transforming our data

        Returns:
            preprocessor object
        '''
        try:
            logger.info('Preprocessor object initializing')
            num_features = X.select_dtypes('number').columns

            logger.info('Building Numerical features pipelin')
            num_pipe = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaling', StandardScaler())
                ]
            )

            logger.info('Building Colum Transformer')
            preprocessor = ColumnTransformer(
                [
                    ('num_preprocessor', num_pipe, num_features)
                ]
            )

            logger.info('Preprocessor object created')

            return preprocessor

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
    def initiate_preprocessing(self, train_path, test_path):
        '''
        Takes path and apply transformation and returns X_train, X_test, y_train, y_test

        Parameters:
            train_path (str): training data path
            test_path (str): testing data path

        Returns:
            X_train, X_test, y_train, y_test
        '''
        
        try:
            logger.info('Initiating Data Preprocessing')
            
            X_train, X_test, y_train, y_test = get_train_test_data(train_path, test_path)

            logger.info('Applying Preprocessing to train and test data')
            preprocessor = self.get_preprocessor_object(X_train)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.data_preprocessing_config.preprocessor_path), exist_ok=True)

            save_model(
                file_path=self.data_preprocessing_config.preprocessor_path,
                model=preprocessor
            )

            logger.info('Data Preprocessing Completed')

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)