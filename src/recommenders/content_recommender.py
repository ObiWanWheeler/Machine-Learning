import pandas as pd
from src.recommenders.recommender import Recommender
import numpy as np
import functools
import operator


class ContentRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame):
        super().__init__(shows, ratings)
        self.genre_frequencies = self.calculate_genre_frequencies()
        self.show_embeddings = self.calculate_item_embeddings()

    def calculate_genre_frequencies(self) -> dict:
        # vectorized approach to iterating over HUGE array. 
        # Does the equivalent of splitting the genre fields on each row by comma,
        # Reducing the dimension of the array to 1,
        # flattening the array.
        genres_non_distinct = functools.reduce(operator.concat, self.shows["genre"].apply(lambda row: row.split(', ')))
        genres_distinct = set(genres_non_distinct)
        return {genre: genres_non_distinct.count(genre) for genre in genres_distinct}

    def calculate_user_embedding(self, user_id) -> dict:
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

    def calculate_item_embeddings(self) -> dict:
        """Calculates ALL item embeddings for item dataframe passed to self. These should never need to be changed.
        """

        embeddings: dict = {}

        show_df_cols = list(self.shows.columns)
        show_tuple_anime_id_index = show_df_cols.index("anime_id")  # index of id column once df is numpy-ified
        show_tuple_genre_index = show_df_cols.index("genre")  # as above but for genre column

        shows_arr = np.array(self.shows)  # convert df to numpy array, as they are far quicker to work with
        for show in shows_arr:  # iterate through each show
            embeddings[show[show_tuple_anime_id_index]] = {}  # initialise this shows embedding
            this_shows_genres = show[show_tuple_genre_index].split(", ")  # extract the show's genres
            for genre in this_shows_genres:  # construct embedding based on genres
                if genre not in embeddings[show[show_tuple_anime_id_index]]:
                    embeddings[show[show_tuple_anime_id_index]][genre] = 1  # if a genre is present it gets a 1
                else:  # fairly sure this never gets entered, but just in case ;)
                    embeddings[show[show_tuple_anime_id_index]][genre] += 1
        return embeddings

    def compare_embeddings(self, user_id) -> dict:
        """Compares user and show embeddings to find show most similar to user.

        Return: dictionary of form show_id: score, highest being most relevant
        """

        # retrieve this user's embedding
        user_embedding = self.get_user_embedding(user_id)

        return {
            show_id: self.calculate_similarity_score(
                user_embedding, show_embedding
            )
            for show_id, show_embedding in self.show_embeddings.items()
        }

    def calculate_similarity_score(self, user_embedding, show_embedding) -> int:
        """Calculates similarity between user embedding and show embedding,
        both given as dictionaries of form genre: score

        Keyword arguments:
        user_embedding -- dictionary of form genre: score for the user. Shows users preferences
        show_embedding -- dictionary of form genre: score for the show. Shows show's genres
        Return: return_description
        """

        # if len(user_embedding.keys()) != len(show_embedding.keys()):
        #     raise Exception("Cannot dot product vectors of different lengths")
        # as it turns out, due to the nature of the data, they may well have different sizes yet still be valid.
        # We need consider only the fields present on both.

        total_score = 0
        for genre, score in show_embedding.items():
            try:
                normalized_score = score / (0.1 * self.genre_frequencies[genre])
                total_score += user_embedding[genre] * normalized_score
            except KeyError:  # user didn't have this genre in their embedding, unlikely but plausible
                pass
        # penalty that prevents shows with loads of genres being recommended too highly
        total_score /= (len(user_embedding) / len(show_embedding))
        return total_score

    def get_user_embedding(self, user_id):
        return (
            self.user_embeddings[user_id]
            # This user's embedding has already been generated and stored, so why bother generating it again?
            if user_id in self.user_embeddings
            else self.calculate_user_embedding(user_id)
        )

    # overridden
    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:

        # do the actual recommending process and figure out the recommendations
        if items_to_ignore is None:
            items_to_ignore = []
        scores = self.compare_embeddings(user_id)
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
