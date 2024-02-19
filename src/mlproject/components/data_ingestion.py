import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.model_tranier import ModelTrainerConfig
from src.mlproject.components.model_tranier import ModelTrainer
from src.mlproject.utils import read_sql_data


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Reading data from MySQL
            df = read_sql_data()
            logging.info("Reading data from the MySQL database completed")

            # Creating directories if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Splitting data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test sets to CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()

    print((model_trainer.initiate_model_trainer(train_data, test_data)))


