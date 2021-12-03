import numpy as np
import pandas as pd

from src.myconstants import PROD
from src.recommenders.recommender import Recommender
from src.utils import calc_mean_squared_error


class CollabRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame) -> None:
        super().__init__(shows, ratings)

        # pivot data from userId - animeId - rating as columns to 
        # userId as rows, animeId as columns, rating as values 
        self.feedback_df = self.ratings.pivot_table(
            index="user_id", columns="anime_id", values="rating", fill_value=0.0, aggfunc=np.mean).astype('float')

        # calculate predictions
        # for now, this only needs to happen once: on startup.
        self.predictions_df = self.init_prediction_df()

    def init_prediction_df(self):
        predictions_matrix = self.calc_sgd_predictions()
        return pd.DataFrame(
            predictions_matrix, index=self.feedback_df.index, columns=self.feedback_df.columns)

    @staticmethod
    def on_epoch_start(epoch_num):
        """
        Development oriented processes to happen at the start of each epoch.
        Mostly logging.
        """
        print(f"Epoch: {epoch_num + 1}")

    # this might expand when we introduce biases
    @staticmethod
    def calc_validity_stats(predictions, actual_data):
        """
        Calculates various error statistics for a prediction
        Parameters
        ------------
        predictions:
            prediction matrix to test for error
        actual_data:
            the original matrix of values
        Returns
        ---------
        tuple containing all the calculated statistics:
            root mean squared error, error stat 2, error stat 3, ...
        """
        return calc_mean_squared_error(predictions[actual_data.nonzero()], actual_data[actual_data.nonzero()])

    def calc_sgd_predictions(self, max_epoch_count=1000, latent_feature_count=75, alpha=0.01, gamma=0.4,
                             accepted_deviation=2.5):
        """Calculates prediction matrix by using matrix factorisation and stochastic gradient descent

        Parameters
        ----------
        max_epoch_count : int > 0
            max number of times to iterate gradient descent to improve prediction accuracy
        latent_feature_count : int
            Number of latent factors to discover - rank to reduce data matrix to.
            The lower this is the faster to execute
        alpha: int
            'learning rate'. Size of steps to take in gradient descent.
            The lower this is the slower to execute
            but the more accurate results that can eventually be obtained.
        gamma: int
            regularization parameter, should be in the range 0.1 - 1.0. Higher values restrict
            model further.
        accepted_deviation:
            the point at which the approximation is 'good enough'. Lower this is the better,
            but the longer it will take.
        ----------
        Returns
        ---------
        populated pivoted dataframe of user id against anime id with predicted ratings as values.
        """

        if max_epoch_count <= 0:
            max_epoch_count = 1000

        # work with numpy arrays instead of dfs
        # demean data to help mitigate any user bias
        feedback_matrix = self.feedback_df.fillna(0.0).to_numpy()

        m, n = feedback_matrix.shape
        user_lfs = np.random.rand(latent_feature_count, m)
        item_lfs = np.random.rand(latent_feature_count, n)

        # the user - rating pairs of ratings that have actually been left
        user, items = feedback_matrix.nonzero()

        predictions = []

        for epoch in range(max_epoch_count):  # iterate gradient descent
            if not PROD:
                self.on_epoch_start(epoch)  # development logging stuff
            predictions = user_lfs.T @ item_lfs  # make an approximation
            # check how good the approximation is
            mse = self.calc_validity_stats(predictions, feedback_matrix)
            if not PROD:
                print("MSE:", mse)

            # check if the approximation is 'good enough'
            if mse < accepted_deviation:
                print(f"broke after {epoch + 1} epochs")
                break

            # if not, iterate gradient descent again and improve our approximation
            for u, i in zip(user, items):
                dot = np.dot(user_lfs[:, u].T, item_lfs[:, i])
                difference = feedback_matrix[u, i] - dot
                user_lfs[:, u] += 2 * alpha * (difference * item_lfs[:, i] - gamma * user_lfs[:, u])
                item_lfs[:, i] += 2 * alpha * (difference * user_lfs[:, u] - gamma * item_lfs[:, i])

        return predictions

    # overridden
    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:
        # find the requested users recommendations
        if items_to_ignore is None:
            items_to_ignore = []
        requested_users_recs = self.predictions_df.loc[user_id]

        # remove any items from items_to_ignore
        filtered_recs = requested_users_recs.filter(items=(set(requested_users_recs.index) - set(items_to_ignore)))

        # sort by recommendation score and give us the top {recommendation_count}
        top_shows = filtered_recs.sort_values(ascending=False).to_frame(name="predicted_rating")[
                    :min(recommendation_count, len(filtered_recs))]

        if verbose:  # add all other show data if requested
            top_shows = top_shows.merge(self.shows[self.shows["anime_id"].isin(
                filtered_recs.keys())], how="left", left_on="anime_id", right_on="anime_id")
        top_shows.reset_index(level=0, inplace=True)
        return top_shows

    def refresh(self):
        self.predictions_df = self.init_prediction_df()

    def get_score_column_name(self) -> str:
        return "predicted_rating"
