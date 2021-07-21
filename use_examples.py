import re
from utils import *
import pandas as pd
from popularity_recommender import PopularityRecommender
from content_recommender import ContentRecommender
from collaborative_recommender import CollabRecommender
from profile_builder import ProfileBuilder
from data import anime, ratings

# data
print(ratings.head())

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

# simple popularity recommender (fastest)
print('Popularity recommender: \n')
pr = PopularityRecommender()
print(
    f'Most popular anime overall:\n {pr.give_recommendation(anime, 10)}', '\n'*3)

# popularity by query
print('Popularity by query: \n')
action_or_adventure_anime = filter_by_query('genre', lambda x: bool(
    re.search(r'Action|Adventure', x, re.IGNORECASE)), anime)
print(
    f'Most popular Action / Adventure anime:\n {pr.give_recommendation(action_or_adventure_anime, 10)}', '\n'*3)


# content recommender (slowest)
pb = ProfileBuilder(anime, item_ids, 'user_id', 'anime_id',
                    'rating', watched_ratings, tfidf_matrix)
user_profiles = pb.build_all_user_profiles()
print('Perceived preferences of user 1: \n')
print(pd.DataFrame(sorted(zip(tfidf_feature_names,
                              user_profiles[1].flatten().tolist()), key=lambda x: -x[1])[:20],
                   columns=['token', 'relevance']), '\n')

print('Content based recommendations for user 1: \n')
cr = ContentRecommender()
recs = cr.give_recommendation(
    user_profiles[1], tfidf_matrix, item_ids, 'anime_id', anime)
print(recs, '\n'*3)

# collab recommender (most individually accurate)
colr = CollabRecommender(watched_ratings, anime,
                         'user_id', 'anime_id', 'rating')
print('Collab based recommendations for user 1: \n')
print(colr.give_recommendations(user_id=1, verbose=True))
