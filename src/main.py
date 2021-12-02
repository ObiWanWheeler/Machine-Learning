import logging
from src.database.data import get_connection_psycopg, DatabaseCustomORM, PsycopCursor
from src.recommender_app import RecommenderApp

logging_format = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logging_format)


# creating FApp and connecting to database
connection = get_connection_psycopg("database/database.ini")
cursor = PsycopCursor(connection.cursor())

db = DatabaseCustomORM(cursor)

app = RecommenderApp(__name__, db)

if __name__ == "__main__":
    app.run()

connection.close()
