import numpy as np
import pandas as pd
import psycopg2
import config
from abc import ABC, abstractmethod


def get_connection_psycopg(ini_file='./database.ini', section='postgresql'):
    """Connects to Postgres server
    Takes the path to an ini file as an argument. 
    This should specify the database host, name, user and password 
    under the section defined by the 'section' parameter.
    """

    conn = None
    try:
        # extracts connection parameters from ini file
        params = config.config(ini_file, section)
        print('connecting to postgres')
        conn = psycopg2.connect(**params)
    # potential for an exception if the ini file is incorrectly formatted, database deemed not to exist... etc.
    except psycopg2.DatabaseError as error:
        print(error)
    finally:
        # If somehow, an error has slipped through the fairly broad "DatabaseError" umbrella, 
        # this will catch any other issues.
        if conn is None:
            raise psycopg2.DatabaseError('Unable to connect to database')
        else:
            return conn


class DBCursor(ABC):

    def __init__(self, cursor):
        self.cursor = cursor

    @abstractmethod
    def execute(self, query: str, params: tuple):
        pass

    @abstractmethod
    def fetch_all(self):
        pass

    @abstractmethod
    def fetch_n(self, n):
        pass

    @abstractmethod
    def close(self):
        pass


class PsycopCursor(DBCursor):

    def execute(self, query: str, params: tuple):
        self.cursor.execute(query % params)

    def fetch_all(self):
        return self.cursor.fetchall()

    def fetch_n(self, n):
        return self.cursor.fetchmany(n)

    def close(self):
        self.cursor.close()


class DatabaseCustomORM:

    def __init__(self, db_cursor: DBCursor):
        self.cursor = db_cursor

    def fetch_all(self, table_name: str):
        self.cursor.execute("SELECT * FROM %s;", (table_name,))
        return self.cursor.fetch_all()

    def fetch_n(self, table_name: str, n: int):
        self.cursor.execute("SELECT * FROM %s LIMIT=%s;", (table_name, n))
        return self.cursor.fetch_n(n)

    def fetch_by_condition(self, table_name: str, condition: str):
        self.cursor.execute("SELECT * FROM %s WHERE %s;", (table_name, condition))
        return self.cursor.fetch_all()

    def __del__(self):
        self.cursor.close()


def fetch_feedback_data(db) -> pd.DataFrame:
    ratings = db.fetch_all("rating")
    ratings_df = pd.DataFrame(
        ratings, columns=['user_id', 'anime_id', 'rating', 'createdAt', 'updatedAt'])
    ratings_df['rating'] = ratings_df['rating'].astype(np.int8)
    return ratings_df[ratings_df['rating'] != 0]


def fetch_anime_data(db) -> pd.DataFrame:
    anime = db.fetch_all("anime")
    anime_df = pd.DataFrame(anime, columns=[
        'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'updatedAt', 'createdAt', 'synopsis',
        'titleImage'])
    anime_df = anime_df[~anime_df["genre"].str.contains("Hentai")]
    anime_df.dropna(inplace=True)

    return anime_df
