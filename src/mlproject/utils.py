import os
import pickle
import sys

import cloudpickle
import pandas as pd
import pymysql
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # Use 'with' statement to ensure the connection is properly closed
        with pymysql.connect(
                host=host,
                user=user,
                password=password,
                db=db
        ) as mydb:
            logging.info('Connection Established')
            # Use parameterized query to prevent SQL injection
            query = 'SELECT * FROM raw'
            df = pd.read_sql_query(query, mydb)

            # Print or log the first few rows of the dataframe
            logging.info(df.head())

            return df
    except Exception as ex:
        logging.error(f"Error: {ex}")
        raise CustomException(ex)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            cloudpickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# Call the function
read_sql_data()


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
