import pandas as pd
import re
import numpy as np
from pandas.core.frame import DataFrame
from content_recommender import ContentRecommender
from popularity_recommender import PopularityRecommender
from profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix, filter_by_query
from scipy.sparse.linalg import svds

# data sets
anime = pd.read_csv("./data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("./data/anime/rating.csv", low_memory=False)

# data cleaning
anime['genre'].fillna('', inplace=True)
anime['type'].fillna('', inplace=True)
anime['name'].fillna('', inplace=True)

# abridging ratings so the ol' laptop can actually run it
ratings = ratings[ratings['user_id'] <= 1000]

# content stuff
item_ids = anime['anime_id'].tolist()
tfidf_feature_names, tfidf_matrix = calculate_tfidf_matrix(anime)

watched_ratings = ratings.loc[ratings['rating'] != -1]

# collaborative stuff
user_ratings = watched_ratings.pivot(
    index='user_id', columns='anime_id', values='rating').fillna(0)
user_ratings_matrix = user_ratings.to_numpy()
user_ratings_means = np.mean(user_ratings_matrix, axis=1)
user_ratings_demeaned = user_ratings_matrix - user_ratings_means.reshape(-1, 1)

u, sigma, vt = svds(user_ratings_demeaned, k=75)
sigma = np.diag(sigma)
predictions_matrix = np.dot(np.dot(u, sigma), vt) + \
    user_ratings_means.reshape(-1, 1)
predictions_df = DataFrame(predictions_matrix, columns=user_ratings.columns)


def give_recommendations(user_id, items_df=None, topn=10):
    user_predictions = predictions_df.iloc[user_id].sort_values(
        ascending=False).reset_index().rename(columns={user_id: 'relevance'})

    user_ratings = watched_ratings[watched_ratings['user_id'] == user_id]

    recommendations = user_predictions[~user_predictions['anime_id'].isin(
        user_ratings['anime_id'])].sort_values(by='relevance', ascending=False).head(topn)

    if items_df is not None:
        recommendations = recommendations.merge(items_df, how='left', left_on='anime_id', right_on='anime_id')

    return recommendations

# pr = PopularityRecommender()
# print(f'Most popular anime overall:\n {pr.give_recommendation(anime, 10)}')

# action_or_adventure_anime = filter_by_query('genre', lambda x: bool(
#     re.search(r'Action|Adventure', x, re.IGNORECASE)), anime)
# print(f'Most popular Action / Adventure anime:\n {pr.give_recommendation(action_or_adventure_anime, 10)}')

# pb = ProfileBuilder(anime, item_ids, 'user_id', 'anime_id', 'rating' ,watched_ratings, tfidf_matrix)
# user_profiles = pb.build_all_user_profiles()
# print(pd.DataFrame(sorted(zip(tfidf_feature_names,
#                               user_profiles[1].flatten().tolist()), key=lambda x: -x[1])[:20],
#                    columns=['token', 'relevance']))
# print(len(user_profiles))
# cr = ContentRecommender()
# recs = cr.give_recommendation(user_profiles[1], tfidf_matrix, item_ids, 'anime_id', anime)
# print(recs)

print(give_recommendations(0, anime))