from recommenders.profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.collaborative_recommender import CollabRecommender
import data
import pandas as pd
import numpy as np


if __name__ == '__main__':
    connection = data.get_connection()
    new_uid = 10001  # dbl.learn_new_user_preferences_db(connection)

    with connection.cursor() as cur:
        cur.execute("SELECT * FROM rating")
        ratings = cur.fetchall()
        ratings_df = pd.DataFrame(
            ratings, columns=['user_id', 'anime_id', 'rating'])
        ratings_df['rating'] = ratings_df['rating'].astype(np.int8)
        watched_ratings = ratings_df[ratings_df['rating'] != 0]

        cur.execute("SELECT * FROM anime")
        anime = cur.fetchall()
        anime_df = pd.DataFrame(anime, columns=[
                                'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'])

        anime_ids = anime_df['anime_id'].to_list()
        tfidf_feature_names, tfidf_matrix = calculate_tfidf_matrix(anime_df)

        items_to_ignore = list(
            watched_ratings[watched_ratings['user_id'] == new_uid]['anime_id'])

        collab = CollabRecommender(
            watched_ratings, user_id_column='user_id', item_id_column='anime_id', rating_column='rating')
        recs = collab.give_recommendations(
            new_uid, anime_df, verbose=True, topn=20, items_to_ignore=items_to_ignore)

        print(f'Collab based recommendations for user {new_uid}: \n')
        print(recs)

        pb = ProfileBuilder(anime_df, anime_ids, 'user_id', 'anime_id',
                            'rating', watched_ratings, tfidf_matrix)
        user_profiles = pb.build_all_user_profiles()
        print(f'Perceived preferences of user {new_uid}: \n')
        print(pd.DataFrame(sorted(zip(tfidf_feature_names,
                                      user_profiles[new_uid].flatten().tolist()), key=lambda x: -x[1])[:20],
                           columns=['token', 'relevance']), '\n')

        cr = ContentRecommender(user_profiles, tfidf_matrix, anime_ids, 'anime_id')
        recs = cr.give_recommendations(
            new_uid, anime_df, verbose=True, topn=20, items_to_ignore=items_to_ignore)
        print(f'Content based recommendations for user {new_uid}: \n')
        print(recs, '\n'*3)

        hr = HybridRecommender(cr, collab, 'anime_id', {
                               'content': 1.0, 'collab': 10.0})
        print(f'Hybrid recommendations for user {new_uid}: \n')
        print(hr.give_recommendations(user_id=new_uid, items_df=anime_df, topn=20,
              verbose=True, items_to_ignore=items_to_ignore)[['anime_id', 'name', 'rating']])

    connection.close()
