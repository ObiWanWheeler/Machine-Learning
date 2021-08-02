from utils import calculate_tfidf_matrix
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.collaborative_recommender import CollabRecommender
import data
import experimenting.db_learning as dbl
import pandas as pd
from profile_builder import ProfileBuilder

if __name__ == '__main__':
    connection = data.get_connection()

    # dbl.learn_new_user_preferences_db(connection)

    with connection.cursor() as cur:
        cur.execute("SELECT * FROM rating")
        ratings = cur.fetchall()
        ratings_df = pd.DataFrame(ratings, columns=['user_id', 'anime_id', 'rating'])
        watched_ratings = ratings_df[ratings_df['rating'] != -1]

        cur.execute("SELECT * FROM anime")
        anime = cur.fetchall()
        anime_df = pd.DataFrame(anime, columns=['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'])

        anime_ids = anime_df['anime_id'].to_list()

        collab = CollabRecommender(watched_ratings, 'user_id', 'anime_id', 'rating')
        
        recs = collab.give_recommendations(73517, anime_df, verbose=True)
        print(recs)

    connection.close()