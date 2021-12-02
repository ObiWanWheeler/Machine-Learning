import pandas as pd

from src.recommenders.collaborative_recommender import CollabRecommender
from src.recommenders.content_recommender import ContentRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.popularity_recommender import PopularityRecommender


class RecommendationEngine:

    def __init__(self, items_df, ratings_df):
        self.items = items_df
        self.feedback = ratings_df

        self.popularity_r = PopularityRecommender()
        self.content_r = ContentRecommender(items_df, ratings_df)
        self.collab_r = CollabRecommender(items_df, ratings_df)
        self.hybrid_r = HybridRecommender(items_df, ratings_df, self.content_r, self.collab_r)

    def refresh_recommenders(self, updated_ratings):
        self.feedback = updated_ratings.copy()
        self.content_r.calculate_item_embeddings()
        predictions_matrix = self.collab_r.calc_sgd_predictions()
        self.collab_r.predictions_df = pd.DataFrame(
            predictions_matrix, index=self.feedback.index, columns=self.feedback.columns)
