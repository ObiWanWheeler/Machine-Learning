from utils import chunk_dataframe
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# one instance per data set


class CollabRecommender:

    def __init__(self, ratings_df: pd.DataFrame, user_id_column, item_id_column, rating_column) -> None:
        self.ratings_df = ratings_df

        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.rating_column = rating_column

        self.user_ids: pd.Series
        self.user_ratings: pd.DataFrame
        self.user_ratings_means: np.ndarray
        self.user_ratings_demeaned: np.ndarray
        self.predictions_df: pd.DataFrame

        self.prep_data()
        self.produce_predictions()

    def prep_data(self):
        # pivoted_ratings = self.ratings_df.pivot_table(index=self.user_id_column,
        #         columns=self.item_id_column,
        #         values=self.rating_column,
        #         fill_value=0,
        #         aggfunc=np.mean)

        # [                                                         Chunking
        #     chunk.pivot_table(
        #         index=self.user_id_column,
        #         columns=self.item_id_column,
        #         values=self.rating_column,
        #         fill_value=0,
        #         aggfunc=np.mean
        #     )
        #     for chunk in chunk_dataframe(self.ratings_df, chunk_size=10**6)
        # ]

        # for c in pivoted_ratings:
        #     print(c)

        # self.user_ratings = pd.concat(pivoted_ratings).fillna(0)

        self.user_ratings = self.ratings_df.pivot_table(index=self.user_id_column,
                columns=self.item_id_column,
                values=self.rating_column,
                fill_value=0,
                aggfunc=np.mean)

        self.user_ids = self.user_ratings.index

        self.user_ratings_matrix = self.user_ratings.to_numpy()
        self.user_ratings_means = np.mean(self.user_ratings_matrix, axis=1) 
        self.user_ratings_demeaned = self.user_ratings_matrix - \
            self.user_ratings_means.reshape(-1, 1)

    def produce_predictions(self):
        u, sigma, vt = svds(self.user_ratings_demeaned, k=75)
        sigma = np.diag(sigma)
        predictions_matrix = np.dot(np.dot(u, sigma), vt) + \
            self.user_ratings_means.reshape(-1, 1)
        self.predictions_df = pd.DataFrame(
            predictions_matrix, index=self.user_ids, columns=self.user_ratings.columns)

    def give_recommendations(self, user_id, items_df, topn=10, verbose=False, items_to_ignore=[]):
        user_predictions = self.predictions_df.loc[user_id].sort_values(
            ascending=False).reset_index().rename(columns={user_id: 'relevance'})

        # print(current_user_ratings.columns)
        recommendations = user_predictions[~user_predictions[self.item_id_column].isin(
            items_to_ignore)].sort_values(by='relevance', ascending=False).head(topn)

        if verbose:
            if items_df is not None:
                recommendations = recommendations.merge(
                    items_df, how='left', left_on=self.item_id_column, right_on=self.item_id_column)
            else:
                raise Exception(
                    'items_df must not be none to use verbose mode')

        return recommendations
