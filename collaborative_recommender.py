import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# one instance per data set
class CollabRecommender:

    def __init__(self, ratings_df, items_df, user_id_column, item_id_column, rating_column) -> None:
        self.ratings_df = ratings_df
        self.items_df = items_df

        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.rating_column = rating_column

        self.user_ratings: pd.DataFrame
        self.user_ratings_means: np.ndarray
        self.user_ratings_demeaned: np.ndarray
        self.predections_df: pd.DataFrame

        self.prep_data()
        self.produce_predictions()

    def prep_data(self):
        self.user_ratings = self.ratings_df.pivot(
            index=self.user_id_column, columns=self.item_id_column, values=self.rating_column).fillna(0)
        self.user_ratings_matrix = self.user_ratings.to_numpy()
        self.user_ratings_means = np.mean(self.user_ratings_matrix, axis=1)
        self.user_ratings_demeaned = self.user_ratings_matrix - \
            self.user_ratings_means.reshape(-1, 1)

    def produce_predictions(self):
        u, sigma, vt = svds(self.user_ratings_demeaned, k=75)
        sigma = np.diag(sigma)
        predictions_matrix = np.dot(np.dot(u, sigma), vt) + \
            self.user_ratings_means.reshape(-1, 1)
        self.predictions_df = pd.DataFrame(predictions_matrix, columns=self.user_ratings.columns)
    
    
    def give_recommendations(self, user_id, topn=10, verbose=False):
        user_id = user_id - 1
        user_predictions = self.predictions_df.iloc[user_id].sort_values(
            ascending=False).reset_index().rename(columns={user_id: 'relevance'})

        user_ratings = self.ratings_df[self.ratings_df[self.user_id_column] == user_id]

        recommendations = user_predictions[~user_predictions[self.item_id_column].isin(
            user_ratings[self.item_id_column])].sort_values(by='relevance', ascending=False).head(topn)

        if verbose:
            recommendations = recommendations.merge(self.items_df, how='left', left_on=self.item_id_column, right_on=self.item_id_column)

        return recommendations
