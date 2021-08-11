import data
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# only needs to be run if the program has never been used on a system (in dev) to initialise relevant databases locally
# requires postgres be installed on system

# data sets
anime = pd.read_csv("./data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("./data/anime/rating.csv", low_memory=False)

# data cleaning
anime['genre'].fillna('Unkown', inplace=True)
anime['type'].fillna('Unknown', inplace=True)
anime['name'].dropna(inplace=True)
anime['episodes'].replace('Unknown', -1, inplace=True)
anime['episodes'].fillna(-2, inplace=True)
anime['rating'].fillna(0.0, inplace=True)

ratings = ratings[ratings['anime_id'].isin(anime['anime_id'])]


# the following functions should only be used on systems that have cloned this repository 
# to transfer the anime data from csv into a local database in dev

def create_tables(connection):
    commands = [
        """
        CREATE TABLE anime (
            anime_id integer PRIMARY KEY,
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
            user_id integer NOT NULL,  
            anime_id integer NOT NULL,
            rating integer NOT NULL CHECK (rating >= 0 AND rating <= 10),
            CONSTRAINT fk_anime
                FOREIGN KEY(anime_id)
                    REFERENCES anime(anime_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
        );
        """
    ]
    
    cursor = connection.cursor()
    for command in commands:
        try:
            cursor.execute(command)
            connection.commit()
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)
            continue
    cursor.close()
    

def populate_tables(connection):
    cursor = connection.cursor()
    try:
        execute_values(cursor, f'INSERT INTO anime (anime_id, name, genre, type, episodes, rating, members) VALUES %s;', anime.values.tolist())
        execute_values(cursor, f'INSERT INTO rating (user_id, anime_id, rating) VALUES %s;', ratings.values.tolist())
    except (psycopg2.DatabaseError) as error:
        print(error)

    connection.commit()
    cursor.close()


# connection = data.get_connection()
# create_tables(connection)
# populate_tables(connection)