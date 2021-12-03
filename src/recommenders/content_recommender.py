import numpy as np
import pandas as pd

from src.recommenders.prediction_algorithms import calculate_similarity_score, calculate_term_frequencies, \
    calculate_item_embeddings
from src.recommenders.recommender import Recommender


class ContentRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame):
        super().__init__(shows, ratings)
        self.genre_frequencies = calculate_term_frequencies(shows, "genre")
        self.show_embeddings = calculate_item_embeddings(shows)

    def __calculate_user_embedding(self, user_id) -> dict:
        """Generates the embedding for the user with id = user_id.\n
        Stores the result in self.user_embeddings[user_id]
        so it need not be generated again and can be updated easily when reviews are left.

        Keyword arguments:\n
        user_id -- the id of the user who's embedding you want to generate
        Return: Calculated user embedding
        """
        embedding = {}

        rating_df_cols = list(self.ratings.columns)
        rating_tuple_anime_id_index = rating_df_cols.index("anime_id")  # index of id column once df is numpy-ified
        rating_tuple_rating_index = rating_df_cols.index("rating")  # as above but for rating column

        # much faster to work with underlying numpy array than the full dataframe, so we convert it here
        this_users_ratings = np.array(
            self.ratings[self.ratings["user_id"] == user_id])

        for rating in this_users_ratings:  # iterate through user's ratings
            # extract genre data
            this_shows_genres = list(
                self.shows.loc[self.shows["anime_id"] == rating[rating_tuple_anime_id_index]]["genre"])[0].split(", ")

            for genre in this_shows_genres:
                if genre not in embedding:  # construct embedding based on genres of the show
                    embedding[genre] = rating[rating_tuple_rating_index]
                else:  # increment this genres score proportional to how much the user liked the show it was on
                    embedding[genre] += rating[rating_tuple_rating_index]
        # normalise embedding based on genre frequencies.
        normalized_embedding = {genre: embedding[genre] / (0.5 * self.genre_frequencies[genre])
                                for genre in embedding.keys() & self.genre_frequencies}

        # store the newly calculated user embedding so it doesn't need to be calculated again.
        self.user_embeddings[user_id] = normalized_embedding
        return embedding

    def __compare_embeddings(self, user_id) -> dict:
        """Compares user and show embeddings to find show most similar to user.

        Return: dictionary of form show_id: score, highest being most relevant
        """

        embedding_scores: dict = {}

        # retrieve this user's embedding
        user_embedding = self.__get_user_embedding(user_id)

        # iterate through shows
        for show_id, show_embedding in self.show_embeddings.items():
            # dot product user and show embedding to produce score for this show

            embedding_scores[show_id] = calculate_similarity_score(
                user_embedding, show_embedding, self.genre_frequencies)
        return embedding_scores

    def __get_user_embedding(self, user_id):
        return (
            self.user_embeddings[user_id]
            # This user's embedding has already been generated and stored, so why bother generating it again?
            if user_id in self.user_embeddings
            else self.__calculate_user_embedding(user_id)
        )

    # overridden
    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:

        # do the actual recommending process and figure out the recommendations
        if items_to_ignore is None:
            items_to_ignore = []
        scores = self.__compare_embeddings(user_id)
        # remove any items from items_to_ignore
        filtered_scores = {sid: score for sid,
                           score in scores.items() if sid not in items_to_ignore}

        # TODO write this sorting algorithm myself
        # sort shows based on their recommendation score and then cast to a DataFrame
        top_shows = pd.DataFrame(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[
                                 :min(recommendation_count, len(filtered_scores))],
                                 columns=["anime_id", "relevance_score"])

        if verbose:  # add all other show data if requested
            top_shows = top_shows.merge(self.shows[self.shows["anime_id"].isin(
                filtered_scores.keys())], how="left", left_on="anime_id", right_on="anime_id")

        return top_shows

    def refresh(self):
        self.user_embeddings.clear()

    def get_score_column_name(self) -> str:
        return "relevance_score"
