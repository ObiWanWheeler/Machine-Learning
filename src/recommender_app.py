import logging
import re

from flask import request, jsonify

from src.database.data import DatabaseCustomORM, fetch_anime_data, fetch_feedback_data, fetch_one_feedback_data
from src.flask_app import FlaskApp
from src.recomender_route import recommender_route
from src.recommenders.collaborative_recommender import CollabRecommender
from src.recommenders.content_recommender import ContentRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.popularity_recommender import PopularityRecommender
from src.recommenders.recommendation_engine import RecommendationEngine
from src.utils import filter_by_query, get_query_vars


class RecommenderApp(FlaskApp):
    # just treat this like the program class in C#, this is the entry point, for all intents and purposes,
    # it's not really a class in the normal Python sense, it's just a container for some functionality
    def __init__(self, name, db: DatabaseCustomORM):
        super().__init__(name)
        self.db = db
        self.items_df = fetch_anime_data(db)
        self.feedback_df = fetch_feedback_data(db)

        content_r = ContentRecommender(self.items_df, self.feedback_df)
        collab_r = CollabRecommender(self.items_df, self.feedback_df)
        hybrid_r = HybridRecommender(self.items_df, self.feedback_df, [(content_r, 1.0), (collab_r, 2.0)])

        self.popularity_r = PopularityRecommender()

        self.engine = RecommendationEngine(self.items_df, self.feedback_df,
                                           recommenders={"content_recommender": content_r,
                                                         "collab_recommender": collab_r,
                                                         "hybrid_recommender": hybrid_r})

    def add_all_endpoints(self):
        self.add_endpoint(endpoint="/", endpoint_name="/", handler=self.index)

        self.add_endpoint(endpoint="/popularity-recommender", endpoint_name="/popularity-recommender",
                          handler=self.popularity_recommender_route)
        self.add_endpoint(endpoint="/content-recommender/<int:user_id>", endpoint_name="/content-recommender",
                          handler=self.content_recommender_route)
        self.add_endpoint(endpoint="/collab-recommender/<int:user_id>", endpoint_name="/collab-recommender",
                          handler=self.collab_recommender_route)
        self.add_endpoint(endpoint="/hybrid-recommender/<int:user_id>", endpoint_name="/hybrid-recommender",
                          handler=self.hybrid_recommender_route)

    def try_update_ratings(self, user_id: int):
        old_user_ratings = self.feedback_df[self.feedback_df["user_id"] == user_id]
        new_user_ratings = fetch_one_feedback_data(self.db, user_id)

        old_user_ratings.reset_index(inplace=True, drop=True)
        new_user_ratings.reset_index(inplace=True, drop=True)

        if not old_user_ratings.equals(new_user_ratings):
            logging.debug("New ratings detected, Refreshing recommenders")
            self.engine.refresh_recommenders(new_user_ratings)

    def index(self):
        return "Welcome to the film recommender API"

    def content_recommender_route(self, user_id: int):
        self.try_update_ratings(user_id)
        return recommender_route(user_id, self.engine.get_recommender("content_recommender"))

    def collab_recommender_route(self, user_id: int):
        self.try_update_ratings(user_id)
        return recommender_route(user_id, self.engine.get_recommender("collab_recommender"))

    def hybrid_recommender_route(self, user_id: int):
        self.try_update_ratings(user_id)
        return recommender_route(user_id, self.engine.get_recommender("hybrid_recommender"))

    def popularity_recommender_route(self):
        recommendation_count, verbose = get_query_vars()

        filtered_shows = None
        if query := request.args.get('query'):
            regex_string = query.replace(',', '|')
            filtered_shows = filter_by_query('genre', lambda x: bool(
                re.search(regex_string, x, re.IGNORECASE)), self.items_df)

        shows = filtered_shows if (filtered_shows is not None) else self.items_df
        recs = self.popularity_r.generate_recommendations(
            shows, recommendation_count, verbose)
        return jsonify({"recommendations": recs.to_dict('records')})
