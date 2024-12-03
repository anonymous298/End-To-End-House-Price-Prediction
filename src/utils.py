import os
import sys
import numpy as np
import pandas as pd
import dill

from src.logger import get_logger
from src.exception import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout    
from tensorflow.keras.metrics import R2Score

logger = get_logger('utils')

def get_train_test_data(train_path: str, test_path: str):
    try:
        logger.info('converting data path into X_train, X_test, y_train, y_test')

        logger.info('Reading training and testing data from path')
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        DEPENDENT_FEATURE = 'price'

        logger.info('Converting data into dependent and independent features')
        X_train, X_test, y_train, y_test = (
            train_data.drop([DEPENDENT_FEATURE],axis=1),
            test_data.drop([DEPENDENT_FEATURE], axis=1),
            train_data[DEPENDENT_FEATURE],
            test_data[DEPENDENT_FEATURE]
        )

        logger.info('Data Splitting Completed')

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
    
def save_model(file_path: str, model):
    '''
    saves the model to desired file path

    Parameters:
        file_path (str): Path where the model have to be saved
        model (model): Model which have to be saved

    Returns:
        None
    '''

    try:
        logger.info('Saving Model to path')
        with open(file_path, 'wb') as file:
            dill.dump(model, file)

        logger.info('Model Saved')
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    
def get_models():
    '''
    returns alot of models

    Returns:
        model dictionary
    '''

    logger.info('Initializing Models')
    models = {
        'LinearRegression' : LinearRegression(),
        'SVR' : SVR(),
        'KNeighborsRegressor' : KNeighborsRegressor(),
        'DecisionTreeRegressor' : DecisionTreeRegressor(),
        'RandomForestRegressor' : RandomForestRegressor(),
        'XGBRegressor' : XGBRegressor()
    }

    return models

def get_neural_network(input_shape: tuple, neurons=64, layers=1, activation='relu', optimizer='adam', loss='mse', dropout=0.2):
    '''
    Returns the Neural Network Architecture Sequential Object
    
    Parameters:
        input_shape : defines the input layer of the Neural Network
        neurons : define the size of neuron in each layer
        layers : how many dense layer should a architecture have
        activation : what activation function to give to each neuron
        optimizer : what optimizer to select for optimizint the weights
        loss = select loss for comparing actual value with predicted value

    Returns:
        Sequential model object
    '''

    try:
        logger.info('Building Neural Network Architecture')
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))

        for i in range(layers):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(dropout))

        model.add(Dense(1))

        r2_score = R2Score(name='r2_score')

        model.compile(optimizer=optimizer, loss=loss, metrics=[r2_score])

        logger.info('Neural Network Created')

        return model

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    
def evaluate_model(actual, pred):
    '''
    evaluate model performance based on actual and predicted value

    Parameters:
        actual: actual label 
        pred: prediction from the model

    Returns:
        different metrics for our model.
    '''
    try:
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)

        return mse, mae, rmse, r2
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def nn_evaluate(model, X, y):
    '''
    Evaluates our Neural Network and gives us the score

    Parameters:
        X : our independent data
        y : our dependent data

    Returns:
        the score of the model
    '''

    try:
        _, X_score = model.evaluate(X, y)

        return X_score
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    
def load_object(file_path: str):
    '''
    Loads and return model from path
    '''
    try:
        logger.info('loading model from path')
        with open(file_path, 'rb') as f:
            return dill.load(f)

    except Exception as e:
        raise CustomException(e, sys)