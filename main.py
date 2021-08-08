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
    # new_uid = dbl.learn_new_user_preferences_db(connection)

    with connection.cursor() as cur:
        cur.execute("SELECT * FROM rating")
        ratings = cur.fetchall()
        ratings_df = pd.DataFrame(ratings, columns=['user_id', 'anime_id', 'rating'])

        cur.execute("SELECT * FROM anime")
        anime = cur.fetchall()
        anime_df = pd.DataFrame(anime, columns=['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'])

        anime_ids = anime_df['anime_id'].to_list()

        collab = CollabRecommender(ratings_df, user_id_column='user_id', item_id_column='anime_id', rating_column='rating')
        
        recs = collab.give_recommendations(20000, anime_df, verbose=True)
        print(recs)

    connection.close()