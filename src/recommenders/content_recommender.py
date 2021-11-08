from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.recommender import Recommender
import numpy as np
from operator import itemgetter


class ContentRecommender(Recommender):

    def __init__(self, shows, ratings: DataFrame):
        super().__init__(shows, ratings)
        self.show_embeddings = self.calculate_item_embeddings()

    def calculate_user_embedding(self, user_id):
        """Generates the embedding for the user with id = user_id.\n
        Stores the result in self.user_embeddings[user_id] so it need not be generated again and can be updated easily when reviews are left.
        
        Keyword arguments:\n
        user_id -- the id of the user who's embedding you want to generate
        Return: None
        """
        
        embedding = {}
        
        rating_df_cols = list(self.ratings.columns)
        rating_tuple_anime_id_index = rating_df_cols.index("anime_id")
        rating_tuple_rating_index = rating_df_cols.index("rating")

        this_users_ratings = np.array(
            self.ratings[self.ratings["user_id"] == user_id])

        for rating in this_users_ratings:
            this_shows_genres = list(
                self.shows.loc[self.shows["anime_id"] == rating[rating_tuple_anime_id_index]]["genre"])[0].split(", ")
            for genre in this_shows_genres:
                if genre not in embedding:
                    embedding[genre] = rating[rating_tuple_rating_index]
                else:
                    embedding[genre] += rating[rating_tuple_rating_index]
        self.user_embeddings[user_id] = embedding
        return embedding

    def calculate_item_embeddings(self):
        embeddings: dict = {}

        show_df_cols = list(self.shows.columns)
        show_tuple_anime_id_index = show_df_cols.index("anime_id")
        show_tuple_genre_index = show_df_cols.index("genre")

        shows_arr = np.array(self.shows)
        for show in shows_arr:
            embeddings[show[show_tuple_anime_id_index]] = {}
            this_shows_genres = show[show_tuple_genre_index].split(", ")
            for genre in this_shows_genres:
                if genre not in embeddings[show[show_tuple_anime_id_index]]:
                    embeddings[show[show_tuple_anime_id_index]][genre] = 1
                else:  # fairly sure this never get's entered, but just in case ;)
                    embeddings[show[show_tuple_anime_id_index]][genre] += 1
        return embeddings

    def compare_embeddings(self, user_id):
        embedding_scores: dict = {}

        if user_id in self.user_embeddings: 
        # This user's embedding has already been generated and stored, so why bother generating it again?
            user_embedding = self.user_embeddings[user_id]
        else:
            user_embedding = self.calculate_user_embedding(user_id)

        for id, embedding in self.show_embeddings.items():
            for genre, score in embedding.items():
                try:
                    embedding_scores[id] = user_embedding[genre] * score
                except(KeyError):  # user didn't have this genre in their embedding, unlikely but plausible
                    pass
        return embedding_scores

    # overriden

    def generate_recommendations(self, user_id: int, recommendation_count: int, verbose: bool, items_to_ignore=[]):
        scores = self.compare_embeddings(user_id)
        top_shows = dict(sorted(scores.items(), key=itemgetter(
            1), reverse=True)[:recommendation_count])

        if verbose:
            recommendations: DataFrame = self.shows[self.shows["anime_id"].isin(
                top_shows.keys())].copy()
            recommendations['score'] = top_shows
            recommendations.sort_values(
                by="score", ascending=False, inplace=True)
        else:
            recommendations = top_shows
        return recommendations
