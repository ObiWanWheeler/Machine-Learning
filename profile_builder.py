import scipy
import sklearn
import numpy as np

class ProfileBuilder:

    def __init__(self, items_df, all_item_ids, ratings_df, tfidf_matrix) -> None:
        self.items_df = items_df
        self.all_item_ids = all_item_ids
        self.ratings_df = ratings_df
        self.tfidf_matrix = tfidf_matrix

    def get_one_item_profile(self, item_id):
        idx = self.all_item_ids.index(item_id)
        return self.tfidf_matrix[idx:idx + 1]

    def get_all_item_profiles(self, item_ids):

        item_profiles = [self.get_one_item_profile(id) for id in item_ids]
        item_profiles = scipy.sparse.vstack(item_profiles)
        return item_profiles

    def build_one_user_profile(self, user_ratings):

        user_item_profiles = self.get_all_item_profiles(user_ratings['anime_id'])

        user_item_weights = np.array(user_ratings['rating']).reshape(-1, 1)
        user_items_weighted_avg = np.sum(user_item_profiles.multiply(
            user_item_weights), axis=0) / np.sum(user_item_weights)

        return sklearn.preprocessing.normalize(
            user_items_weighted_avg)

    def build_all_user_profiles(self):
        ratings_indexed = self.ratings_df[self.ratings_df['anime_id'].isin(
            self.items_df['anime_id'])].set_index('user_id')

        return {
            user_id: self.build_one_user_profile(ratings_indexed[ratings_indexed.index == user_id])
            for user_id in ratings_indexed.index.unique()
        }