import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# only needs to be run if the program has never been used on a system (in dev) to initialise relevant databases locally
# requires postgres be installed on system

# data sets
anime = pd.read_csv("../data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("../data/anime/rating.csv", low_memory=False)

# data cleaning
anime['genre'].fillna('Unknown', inplace=True)
anime['type'].fillna('Unknown', inplace=True)
anime['name'].dropna(inplace=True)
anime['episodes'].replace('Unknown', -1, inplace=True)
anime['episodes'].fillna(-2, inplace=True)
anime['rating'].fillna(0.0, inplace=True)

ratings = ratings[ratings['anime_id'].isin(anime['anime_id'])]


# the following functions should only be used on systems that have cloned this repository 
# to transfer the anime data from csv into a local database in dev

def create_tables(connection):
    """Creates anime and rating tables"""

    commands = [
        """
        CREATE TABLE anime (
            "animeId" integer PRIMARY KEY,
            name varchar(100) NOT NULL,
            genre varchar(200) NOT NULL,
            type varchar(20) NOT NULL,
            episodes integer NOT NULL CHECK ((episodes >= -2) and (episodes <> 0)),
            rating float NOT NULL CHECK (rating >= 0.0 AND rating <= 10.0),  
            members integer NOT NULL
        );
        """,
        """ 
        CREATE TABLE rating (
            "userId" integer NOT NULL,  
            "animeId" integer NOT NULL,
            rating integer NOT NULL CHECK (rating >= 0 AND rating <= 10),
            CONSTRAINT fk_anime
                FOREIGN KEY(anime_id)
                    REFERENCES anime(anime_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE "user" (
            id integer NOT NULL PRIMARY KEY,
            username VARCHAR(50) NOT NULL CHECK (LENGTH(username) >= 3),
            email VARCHAR(50) CHECK (email LIKE '%_@__%.__%'),
            password VARCHAR(50)
                CHECK (LENGTH(password) >= 4)
                AND REGEXP_LIKE(password, '(?=\d)(?=^[a-zA-Z0-9]*$)(?=[A-Z])(?=[a-z])')
        )
        """
    ]

    cursor = connection.cursor()
    # trys to create the tables, if an error occurs it prints it and carries on.
    for command in commands:
        try:
            cursor.execute(command)
            connection.commit()
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)
            continue
    # close database cursor to save dat memory
    cursor.close()


def populate_tables(connection):
    """Transfers data from csvs into database tables"""

    cursor = connection.cursor()
    # here we use the psycopg python library to simplify executing the same SQL statement many times over
    try:
        # add all of the anime loaded from csv into the anime table
        execute_values(
            cursor,
            'INSERT INTO anime ("animeId", name, genre, type, episodes, rating, members) VALUES %s;',
            anime.values.tolist(),
        )
        # add all of the ratings loaded from csv into the rating table
        execute_values(
            cursor,
            'INSERT INTO rating ("userId", "animeId", rating) VALUES %s',
            ratings.values.tolist(),
        )

    except psycopg2.DatabaseError as error:
        print(error)

    # save the changes into the database
    connection.commit()
    cursor.close()

# connection = data.get_connection()
# create_tables(connection)
# populate_tables(connection)
