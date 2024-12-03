import os
import sys

from src.exception import CustomException
from src.logger import get_logger
from src.utils import get_models, get_neural_network

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

logger = get_logger('model-trainer')

class ModelTrainer():
    def __init(self):
        pass

    def start_training(self, X_train, X_test, y_train, y_test):
        '''
        Starts Training of different models

        Parameters:
            X_train: X_Train data
            X_test: X_test data
            y_train: y_train data
            y_test: y_test data

        Returns:
            trained models ready to be evaluated.
            Neural Network model
        '''
        trained_models = {}

        try:
            models = get_models()
            logger.info("Starting Model Training")
            for model_name, model in models.items():
                model = model
                model.fit(X_train, y_train)
                logger.info(f'Model Trained: {model_name}')

                trained_models[model_name] = model

            logger.info("All models trained")

            neural_network = get_neural_network((X_train.shape[1],), 128, 3, 'relu', 'adam', 'mse', 0.2)

            logger.info('Training our Neural Network')

            es_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            tb_callback = TensorBoard(log_dir='NN_Logs/training3', histogram_freq=1)

            neural_network.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[es_callback, tb_callback])

            logger.info('Neural Network Trained')

            return (
                trained_models,
                neural_network
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)