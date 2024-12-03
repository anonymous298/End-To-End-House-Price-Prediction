import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object

from tensorflow.keras.models import load_model

logger = get_logger('prediction pipeline')

@dataclass 
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')
    model_path: str = os.path.join('model', 'model.keras')

class CustomData:
    def __init__(self,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,total_rooms):
        self.sqft_living = sqft_living
        self.sqft_lot = sqft_lot
        self.floors = floors
        self.waterfront = waterfront
        self.view = view
        self.condition = condition
        self.grade = grade
        self.sqft_above = sqft_above
        self.sqft_basement = sqft_basement
        self.yr_built = yr_built
        self.yr_renovated = yr_renovated
        self.total_rooms = total_rooms

    def conver_data_to_dataframe(self):
        '''
        Returns the dataframe from inputs the user gave
        '''
        try:
            logger.info('Converting inputs to dataframe')
            data = {
                "sqft_living": [self.sqft_living],
                "sqft_lot": [self.sqft_lot],
                "floors": [self.floors],
                "waterfront": [self.waterfront],
                "view": [self.view],
                "condition": [self.condition],
                "grade": [self.grade],
                "sqft_above": [self.sqft_above],
                "sqft_basement": [self.sqft_basement],
                "yr_built": [self.yr_built],
                "yr_renovated": [self.yr_renovated],
                "total_rooms" : [self.total_rooms]
            }

            return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self, input_df):
        '''
        Predicts the output from the inputs the user give

        Parameters:
            input_df: input dataframe.

        Returns:
            model Prediction.
        '''
        logger.info('Started taking prediction')

        try:

            preprocessor = load_object(self.prediction_pipeline_config.preprocessor_path)
            model = load_model(self.prediction_pipeline_config.model_path)

            logger.info("Applying transformation and prediction")
            X = preprocessor.transform(input_df)
            prediction = float(model.predict(X)[0])

            logger.info('Prediction Complete')

            return prediction

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)