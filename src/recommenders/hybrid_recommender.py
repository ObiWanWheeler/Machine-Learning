from typing import List, Tuple

import pandas as pd

from .recommender import Recommender
from ..exceptions import DataMatchError


class HybridRecommender(Recommender):
    # combines the functionality of several different recommenders, instances of which are injected into this class
    # recommenders passed as (recommender instance, weighting) pairs.
    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame,
                 recommenders: List[Tuple[Recommender, float]]) -> None:
        super().__init__(shows, ratings)

        for recommender, weighting in recommenders:
            if not (recommender.shows.equals(shows) and recommender.ratings.equals(ratings)):
                raise DataMatchError("All recommenders must be trained on the passed "
                                     "items / feedback dataset to "
                                     "be used in conjunction in a hybrid recommender")

        self.recommenders = recommenders

    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:
        if items_to_ignore is None:
            items_to_ignore = []

        joint_recs_df = pd.DataFrame(columns=["anime_id", "joint_score"])
        # execute the recommendation algorithm of each recommender passed to this object
        # and merge their results into one combined dataframe
        for recommender, weighting in self.recommenders:
            recs = recommender.generate_recommendations(user_id, -1, False, [])
            score_column = recommender.get_score_column_name()
            recs[score_column + "_normalized"] = 10 * (recs[score_column] / max(recs[score_column]))

            joint_recs_df = pd.merge(left=joint_recs_df, right=recs, how='outer', left_on='anime_id',
                                     right_on='anime_id').fillna(0.0)
            # contribute to each shows score in proportion to this recommenders weighting
            joint_recs_df['joint_score'] += recs[score_column + "_normalized"] * weighting

        # filter...
        filtered_recs = joint_recs_df.filter(items=(set(joint_recs_df.index) - set(items_to_ignore)), axis=0)
        # and then sort
        top_shows = filtered_recs.sort_values(by='joint_score', ascending=False)[
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
