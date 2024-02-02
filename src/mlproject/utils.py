import os
import pandas as pd
import pymysql
from dotenv import load_dotenv
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


# Call the function
read_sql_data()
