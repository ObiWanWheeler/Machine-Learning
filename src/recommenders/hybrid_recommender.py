from .recommender import Recommender
from .collaborative_recommender import CollabRecommender
from .content_recommender import ContentRecommender
import pandas as pd


class HybridRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame, content_recommender: ContentRecommender,
                 collaborative_recommender: CollabRecommender,
                 weights: dict = {'collab': 10.0, 'content': 1.0}) -> None:
        super().__init__(shows, ratings)

        self.cont_r = content_recommender
        self.collab_r = collaborative_recommender
        self.weights = weights

    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:
        # top content based recommendations
        if items_to_ignore is None:
            items_to_ignore = []
        cont_recs_df = self.cont_r.generate_recommendations(
            user_id, len(self.shows), verbose, items_to_ignore)

        # top collab based recommendations
        collab_recs_df = self.collab_r.generate_recommendations(
            user_id, len(self.shows), verbose, items_to_ignore)

        # merge results into one frame...
        joint_recs_df = pd.merge(left=cont_recs_df, right=collab_recs_df,
                                 how='outer', left_on="anime_id", right_on="anime_id").fillna(0.0)
        # calculate score based on weightings specified...
        normalized_relevance_scores = 10 * (joint_recs_df['relevance_score'] / max(joint_recs_df["relevance_score"]))
        joint_recs_df['joint_relevance_score'] = (normalized_relevance_scores * self.weights['content']) + (
                    joint_recs_df['predicted_rating'] * self.weights['collab'])

        # and sort 
        top_shows = joint_recs_df.sort_values(by='joint_relevance_score', ascending=False)[
                    :min(recommendation_count, len(joint_recs_df))]

        if verbose:
            top_shows = top_shows.merge(
                self.shows, how='left', left_on="anime_id", right_on="anime_id")

        return top_shows
