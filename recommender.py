import pandas as pd
import re
import numpy as np
from pandas.core.frame import DataFrame
from content_recommender import ContentRecommender
from popularity_recommender import PopularityRecommender
from profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix, filter_by_query
from scipy.sparse.linalg import svds
from collaborative_recommender import CollabRecommender

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



