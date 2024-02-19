import sys
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.csv')


def get_data_transformer_object():
    try:

        # Define numerical and categorical features
        numerical_features = ['AGE']
        categorical_features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING',
                                'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

        # Create a custom transformer to apply LabelEncoder to specified columns
        def label_encode_columns(data, columns):
            le = LabelEncoder()
            for column in columns:
                data[column] = le.fit_transform(data[column])
            return data

        # Create a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    #  ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', FunctionTransformer(label_encode_columns, kw_args={'columns': categorical_features}),
                 categorical_features)
            ]
        )

        # Create the final pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('to_integer', FunctionTransformer(lambda x: x.astype(int), validate=False))  # Convert to integers
            # Add more steps as needed
        ])

        # Apply the pipeline to your data
        # df_preprocessed = full_pipeline.fit_transform()
        # df = pd.DataFrame(df_preprocessed)
        # # Display the preprocessed data
        # print(df)
        return full_pipeline

    except Exception as e:
        raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = get_data_transformer_object()
            target_column_name = "LUNG_CANCER"
            numerical_columns = ["AGE"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info((pd.DataFrame(train_arr)).head())
            # print(pd.DataFrame(train_arr))
            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
