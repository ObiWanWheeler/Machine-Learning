import scipy
import sklearn
import numpy as np

class ProfileBuilder:

    def __init__(self, items_df, all_item_ids, user_id_column, item_id_column, rating_column, ratings_df, tfidf_matrix) -> None:
        self.items_df = items_df
        self.all_item_ids = all_item_ids
        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.rating_column = rating_column
        self.ratings_df = ratings_df
        self.ratings_df[self.ratings_df['rating'] < 0]['rating'] = 0
        self.tfidf_matrix = tfidf_matrix

    def get_one_item_profile(self, item_id):
        idx = self.all_item_ids.index(item_id)
        return self.tfidf_matrix[idx:idx + 1]

    def get_all_item_profiles(self, item_ids):

        item_profiles = [self.get_one_item_profile(id) for id in item_ids]
        item_profiles = scipy.sparse.vstack(item_profiles)
        return item_profiles

    def build_one_user_profile(self, user_ratings):
        
        user_item_profiles = self.get_all_item_profiles(user_ratings[self.item_id_column])

        user_item_weights = np.array(user_ratings[self.rating_column]).reshape(-1, 1)
        weight_sum = np.sum(user_item_weights)

        user_items_weighted_avg = np.sum(user_item_profiles.multiply(
            user_item_weights), axis=0)

        if weight_sum != 0:
            user_items_weighted_avg /= weight_sum

        return sklearn.preprocessing.normalize(
            user_items_weighted_avg)

    def build_all_user_profiles(self):
        ratings_indexed = self.ratings_df[self.ratings_df[self.item_id_column].isin(
            self.items_df[self.item_id_column])].set_index(self.user_id_column)

        return {
            user_id: self.build_one_user_profile(ratings_indexed[ratings_indexed.index == user_id])
            for user_id in ratings_indexed.index.unique()
        }