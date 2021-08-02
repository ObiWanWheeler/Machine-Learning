import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import config


def get_connection(ini_file='database.ini', section='postgresql'):
    """Connects to Postgres server"""
    
    conn = None
    try:
        params = config.config(ini_file, section)

        print('connecting to postgres')
        conn = psycopg2.connect(**params)

    except (psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is None:
            raise Exception('Unable to connect to database')
        else:
            return conn





