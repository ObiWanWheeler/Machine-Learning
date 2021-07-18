import numpy as np
from pandas.core.arrays.categorical import contains
from pandas.core.frame import DataFrame
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re
from popularity_recommender import PopularityRecommender
from utils import filter_by_query

# data sets
anime = pd.read_csv("./data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("./data/anime/rating.csv", low_memory=False)

# data cleaning
anime['genre'].fillna('', inplace=True)
anime['type'].fillna('', inplace=True)
anime['name'].fillna('', inplace=True)

# content stuff
watched_ratings = ratings.loc[ratings['rating'] != -1]

vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 2),
                             min_df=0.003,
                             max_df=0.5,
                             max_features=5000,
                             stop_words='english')
item_ids = anime['anime_id'].tolist()
tfidf_matrix = vectorizer.fit_transform(
    anime['name'] + "" + anime['genre'] + "" + anime['type'])
tfidf_feature_names = vectorizer.get_feature_names()



def get_one_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile


def get_all_item_profiles(item_ids):

    item_profiles = [get_one_item_profile(id) for id in item_ids]
    item_profiles = scipy.sparse.vstack(item_profiles)
    return item_profiles


def build_one_user_profile(user_id, ratings_indexed):
    this_users_ratings = ratings_indexed.loc[[user_id]]

    user_item_profiles = get_all_item_profiles(this_users_ratings['anime_id'])

    user_item_weights = np.array(this_users_ratings['rating']).reshape(-1, 1)
    user_items_weighted_avg = np.sum(user_item_profiles.multiply(
        user_item_weights), axis=0) / np.sum(user_item_weights)

    user_profile_normalised = sklearn.preprocessing.normalize(
        user_items_weighted_avg)
    return user_profile_normalised


def build_all_user_profiles():
    ratings_indexed = watched_ratings[watched_ratings['anime_id'].isin(
        anime['anime_id'])].set_index('user_id')
    user_profiles = {}

    for user_id in ratings_indexed.index.unique():
        user_profiles[user_id] = build_one_user_profile(
            user_id, ratings_indexed)
    return user_profiles


user_profiles = build_all_user_profiles()
print(user_profiles)
print(pd.DataFrame(sorted(zip(tfidf_feature_names,
                        user_profiles[-1].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance']))




class ContentRecommender:

    def __init__(self, df) -> None:
        self.df = df

    def give_recommendation(self, topn):
        next


pr = PopularityRecommender()
print(pr.give_recommendation(anime, 10))

action_or_adventure_anime = filter_by_query('genre', lambda x: bool(
    re.search(r'Action|Adventure', x, re.IGNORECASE)), anime)
print(action_or_adventure_anime.head(10))
