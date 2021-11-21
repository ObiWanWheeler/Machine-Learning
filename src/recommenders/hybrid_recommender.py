from recommenders.recommender import Recommender
from .collaborative_recommender import CollabRecommender
from .content_recommender import ContentRecommender
import pandas as pd


class HybridRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame, content_recommender: ContentRecommender, collaborative_recommender: CollabRecommender, weights: dict = {'collab': 10.0, 'content': 1.0}) -> None:
        super().__init__(shows, ratings)

        self.cont_r = content_recommender
        self.collab_r = collaborative_recommender
        self.weights = weights

    def generate_recommendations(self, user_id: int, recommendation_count: int=10, verbose: bool=False, items_to_ignore=[]) -> pd.DataFrame:
        # top content based recommendations
        cont_recs_df = self.cont_r.generate_recommendations(
            user_id, len(self.shows), verbose, items_to_ignore)

        # top collab based recommendatoins
        collab_recs_df = self.collab_r.generate_recommendations(
            user_id, len(self.shows), verbose, items_to_ignore)

        # merge results into one frame
        joint_recs_df = pd.merge(left=cont_recs_df, right=collab_recs_df,
                                 how='outer', left_on="anime_id", right_on="anime_id").fillna(0.0)
        joint_recs_df['joint_relevance_score'] = 10 * ((joint_recs_df['relevance_score'] / max(
            joint_recs_df["relevance_score"])) * self.weights['content']) + (joint_recs_df['predicted_rating'] * self.weights['collab'])

        top_shows = joint_recs_df.sort_values(by='joint_relevance_score', ascending=False)[
            :min(recommendation_count, len(joint_recs_df))]

        if verbose:
            top_shows = top_shows.merge(
                self.shows, how='left', left_on="anime_id", right_on="anime_id")

        return top_shows
