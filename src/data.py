import psycopg2
from psycopg2.extras import execute_values
import config


def get_connection(ini_file='./database.ini', section='postgresql'):
    """Connects to Postgres server
    Takes the path to an ini file as an argument. 
    This should specify the database host, name, user and password 
    under the section defined by the 'section' parameter.
    """
    
    conn = None
    try:
        params = config.config(ini_file, section)
        print('connecting to postgres')
        conn = psycopg2.connect(**params)
    # potential for an exception if the ini file is incorrectly formatted, database deemed not to exist... etc.
    except (psycopg2.DatabaseError) as error: 
        print(error)
    finally:
        # If somehow, an error has slipped through the fairly broad "DatabaseError" umbrella, 
        # this will catch any other issues.
        if conn is None:
            raise Exception('Unable to connect to database')
        else:
            return conn





