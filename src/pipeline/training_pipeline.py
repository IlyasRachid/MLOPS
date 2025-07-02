from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
import sys
from src.logger import logging
from src.exception import CustomException


def run_training_pipeline():
    try:
        logging.info("Starting training pipeline...")

        # Step1: Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path, train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Raw data: {raw_data_path}, Train data: {train_data_path}, Test data: {test_data_path}")

        #Step2: Data Transformation
        data_transformation = DataTransformation()
        transformed_data_path = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
        logging.info(f"Data Transformation completed. Transformed data path: {transformed_data_path}")

        # Step3: Model Training
        model_trainer = ModelTrainer(data_path=transformed_data_path)
        model_path = model_trainer.train_and_save_model()
        logging.info(f"Model Training completed. Model saved at: {model_path}")

        logging.info("Training pipeline completed successfully.")
        return model_path
    
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    try:
        model_path = run_training_pipeline()
        print(f"Model trained and saved at: {model_path}")
    except Exception as e:
        CustomException(e, sys)