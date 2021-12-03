from typing import List, Tuple

import pandas as pd

from .recommender import Recommender


class HybridRecommender(Recommender):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame,
                 recommenders: List[Tuple[Recommender, float]]) -> None:
        super().__init__(shows, ratings)

        self.recommenders = recommenders

    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:
        if items_to_ignore is None:
            items_to_ignore = []

        joint_recs_df = pd.DataFrame(columns=["anime_id", "joint_score"])
        for recommender, weighting in self.recommenders:
            recs = recommender.generate_recommendations(user_id, -1, False, [])
            print(recs)
            score_column = recommender.get_score_column_name()
            recs[score_column + "_normalized"] = 10 * (recs[score_column] / max(recs[score_column]))

            joint_recs_df = pd.merge(left=joint_recs_df, right=recs, how='outer', left_on='anime_id',
                                     right_on='anime_id').fillna(0.0)
            joint_recs_df['joint_score'] += recs[score_column + "_normalized"] * weighting

        # and sort
        top_shows = joint_recs_df.sort_values(by='joint_score', ascending=False)[
                    :min(recommendation_count, len(joint_recs_df))]

        if verbose:
            top_shows = top_shows.merge(
                self.shows, how='left', left_on="anime_id", right_on="anime_id")

        return top_shows

    def refresh(self):
        for recommender, _ in self.recommenders:
            recommender.refresh()

    def get_score_column_name(self) -> str:
        return "joint_score"
