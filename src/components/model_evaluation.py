import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_model, evaluate_model, nn_evaluate

logger = get_logger('model-evaluation')

@dataclass 
class ModelEvaluationPaths:
    model_path: str = os.path.join('model', 'model.pkl')
    nn_model_path: str = os.path.join('model', 'model.keras')

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_paths = ModelEvaluationPaths()

    def evaluate(self, trained_models, nn_model, X_train, X_test, y_train, y_test):
        '''
        Evaluates each model and saves the best model

        Parameters:
            trained_models : Gets trained ML models
            nn_model : trained neural network model
            X_train : X training data
            X_test : X testing data
            y_train : y training data
            y_test : y testing data

        Returns:
            None.
            prints training and testing score of models.
            save the model
        '''

        training_score = {}
        testing_score = {}

        try:
            logger.info('Started evaluating models')   
            for model_name, model in trained_models.items():
                logger.info(f'Evaluating Model -> {model_name}')
                model = model

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_mse, train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
                test_mse, test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

                logger.info(f'Model: {model_name} training r2_score -> {train_r2}')
                logger.info(f'Model: {model_name} testing r2_score -> {test_r2}')

                training_score[model_name] = train_r2
                testing_score[model_name] = test_r2

                logger.info('Evalution completed')

            logger.info('Evaluating our Neural Network model')
            nn_train_score = nn_evaluate(nn_model, X_train, y_train)
            nn_test_score = nn_evaluate(nn_model, X_test, y_test)

            logger.info(f'Neural Network training score -> {nn_train_score}')
            logger.info(f'Neural Network test score -> {nn_test_score}')

            training_score['Neural Network'] = nn_train_score
            testing_score['Neural Network'] = nn_test_score

            logger.info('All evaluation completed')

            logger.info('Saving our best model')

            best_model_score = max(sorted(list(testing_score.values())))
            best_model_name = list(testing_score.keys())[
                list(testing_score.values()).index(best_model_score)
            ]

            logger.info(f'Best Model is -> {best_model_name}')

            if best_model_score <= 0.6:
                raise CustomException('Best model not found', sys)

            if best_model_name == 'Neural Network':
                logger.info('Saving Neural Network')
                nn_model.save(self.model_evaluation_paths.nn_model_path)
                logger.info('Neural Network Saved')

                return None

            best_model = trained_models[best_model_name]

            save_model(
                file_path=self.model_evaluation_paths.model_path,
                model=best_model
            )

            logger.info('Everything completed')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

