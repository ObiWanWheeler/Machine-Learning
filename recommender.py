import pandas as pd
import re
from content_recommender import ContentRecommender
from popularity_recommender import PopularityRecommender
from profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix, filter_by_query

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
