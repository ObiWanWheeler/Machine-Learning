from abc import ABC, abstractmethod

class Recommender(ABC):

    def __init__(self, shows, ratings):
        self.shows = shows
        self.ratings = ratings
        self.item_embeddings: dict = {}
        self.user_embeddings: dict = {}
    
    @abstractmethod
    def generate_recommendations(self, user_id: int, recommendation_count: int, verbose: bool, items_to_ignore = []):
        pass
