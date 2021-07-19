import numpy as np
from pandas.core.arrays.categorical import contains
from pandas.core.frame import DataFrame
import scipy
import pandas as pd
import math
import random
import sklearn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import re
from popularity_recommender import PopularityRecommender
from utils import filter_by_query, calculate_tfidf_matrix
from profile_builder import ProfileBuilder
from content_recommender import ContentRecommender


# data sets
anime = pd.read_csv("./data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("./data/anime/rating.csv", low_memory=False)

# data cleaning
anime['genre'].fillna('', inplace=True)
anime['type'].fillna('', inplace=True)
anime['name'].fillna('', inplace=True)

# abridging ratings so the ol' laptop can actually run it
ratings = ratings[ratings['user_id'] <= 500]

# content stuff
item_ids = anime['anime_id'].tolist()
tfidf_feature_names, tfidf_matrix = calculate_tfidf_matrix(anime)

watched_ratings = ratings.loc[ratings['rating'] != -1]


# pr = PopularityRecommender()
# print(pr.give_recommendation(anime, 10))

# action_or_adventure_anime = filter_by_query('genre', lambda x: bool(
#     re.search(r'Action|Adventure', x, re.IGNORECASE)), anime)
# print(action_or_adventure_anime.head(10))

pb = ProfileBuilder(anime, item_ids, watched_ratings, tfidf_matrix)
user_profiles = pb.build_all_user_profiles()
print(pd.DataFrame(sorted(zip(tfidf_feature_names,
                              user_profiles[2].flatten().tolist()), key=lambda x: -x[1])[:20],
                   columns=['token', 'relevance']))
cr = ContentRecommender()
recs = cr.give_recommendation(user_profiles[2], tfidf_matrix, item_ids, anime)
print(recs)