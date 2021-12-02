from abc import ABC, abstractmethod
import pandas as pd


class Recommender(ABC):

    def __init__(self, shows: pd.DataFrame, ratings: pd.DataFrame):
        self.shows: pd.DataFrame = shows
        self.ratings: pd.DataFrame = ratings
        self.show_embeddings: dict = {}
        self.user_embeddings: dict = {}

    @abstractmethod
    def generate_recommendations(self, user_id: int, recommendation_count: int = 10, verbose: bool = False,
                                 items_to_ignore=None) -> pd.DataFrame:
        """abstract method to be implemented in subclasses. 
        For a given user, finds their recommendations using the implemented algorithm\n
        
        Parameters
        ----------
        user_id -- the id of the user's who recommendations you want to find\n
        recommendation_count -- the number of recommendations you want to be given\n
        verbose -- if False only item ids returned, if True, entire item is returned\n
        items_to_ignore -- list containing item ids of items that shouldn't be recommended\n
        """
        pass
