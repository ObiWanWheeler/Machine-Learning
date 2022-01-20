import logging
from typing import Dict

from src.exceptions import DimensionError, DataMatchError, DuplicateKeyError
from src.recommenders.recommender import Recommender


class RecommendationEngine:

    def __init__(self, items, feedback, recommenders: Dict[str, Recommender]):
        """
        @param items: the items all the recommenders are trained on
        @param feedback: the feedback all the recommenders are trained on
        @param recommenders: a name: instance dictionary of all the recommenders to aggregate
        """
        # otherwise, what's the point
        if len(recommenders) <= 1:
            raise DimensionError("Must provide at least 2 recommenders")

        # make sure all recommenders are trained on the same dataset
        for recommender in recommenders.values():
            if not (recommender.shows.equals(items) and recommender.ratings.equals(feedback)):
                raise DataMatchError("All recommenders must be trained on the passed "
                                     "items / feedback dataset to "
                                     "be used in conjunction")
        self.recommenders = recommenders

    def refresh_recommenders(self, updated_ratings):
        """
        refreshes all recommenders to align with newly inserted ratings
        """
        for recommender in self.recommenders.values():
            recommender.ratings = updated_ratings
            recommender.refresh()

    def get_recommender(self, recommender_name):
        try:
            recommender = self.recommenders[recommender_name]
        except KeyError:
            logging.error(f"Requested recommender {recommender_name} does not exist in this engine!")
        else:
            return recommender

