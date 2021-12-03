import numpy as np
import pandas as pd

from src.recommenders.prediction_algorithms import calc_sgd_predictions
from src.recommenders.recommender import Recommender


class CollabRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame) -> None:
        super().__init__(shows, ratings)

        # pivot data from userId - animeId - rating as columns to 
        # userId as rows, animeId as columns, rating as values 
        self.feedback_df = self.ratings.pivot_table(
            index="user_id", columns="anime_id", values="rating", fill_value=0.0, aggfunc=np.mean).astype('float')

        # calculate predictions
        # for now, this only needs to happen once: on startup.
        self.predictions_df = self.__init_prediction_df()

    def __init_prediction_df(self, **kwargs):
        predictions_matrix = calc_sgd_predictions(self.feedback_df, **kwargs)
        return pd.DataFrame(
            predictions_matrix, index=self.feedback_df.index, columns=self.feedback_df.columns)

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
        self.predictions_df = self.__init_prediction_df(alpha=0.02)

    def get_score_column_name(self) -> str:
        return "predicted_rating"
