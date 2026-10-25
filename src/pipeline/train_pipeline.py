import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """
        Orchestrates the full training pipeline:
        1. Data Ingestion  -> splits raw data into train/test CSVs
        2. Data Transformation -> fits preprocessor and transforms arrays
        3. Model Training  -> grid-searches models, saves best model
        Returns the R² score of the best model on the test set.
        """
        try:
            logging.info(">>> Training Pipeline Started <<<")

            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f">>> Training Pipeline Completed. Best model R² Score: {r2_score:.4f} <<<")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"\nTraining complete! Best model R² Score: {score:.4f}")
