from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger('training-pipelin')

def main():
    '''
    Initializes the training pipeline
    '''
    
    logger.info('Training Pipeline Started')
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.load_data() # Loads the training and testing path

    data_preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = data_preprocessing.initiate_preprocessing(train_path, test_path) # Loads the X_train, X_test, y_train, y_test data.

    model_trainer = ModelTrainer()
    trained_models, trained_nn_model = model_trainer.start_training(X_train, X_test, y_train, y_test) # Returns Trained ML model and Neural Network model

    model_evalution = ModelEvaluation()
    model_evalution.evaluate(trained_models, trained_nn_model, X_train, X_test, y_train, y_test) # Evaluates and save the best model.

    logger.info('Training Pipeline ended.')

if __name__ == '__main__':
    main()