import logging
import re

from flask import request, jsonify

from src.database.data import DatabaseCustomORM, fetch_anime_data, fetch_feedback_data
from src.flask_app import FlaskApp
from src.recomender_route import recommender_route
from src.recommenders.recommendation_engine import RecommendationEngine
from src.utils import filter_by_query


class RecommenderApp(FlaskApp):

    def __init__(self, name, db: DatabaseCustomORM):
        super().__init__(name)
        self.db = db
        self.engine = RecommendationEngine(fetch_anime_data(self.db), fetch_feedback_data(self.db))

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

    def try_update_ratings(self):
        feedback_df_temp = fetch_feedback_data(self.db)

        if not feedback_df_temp.equals(self.engine.feedback):
            logging.debug("New ratings detected, Refreshing recommenders")
            self.engine.refresh_recommenders(feedback_df_temp)

    def index(self):
        return "Welcome to the film recommender API"

    def content_recommender_route(self, user_id: int):
        self.try_update_ratings()
        return recommender_route(user_id, self.engine.content_r)

    def collab_recommender_route(self, user_id: int):
        self.try_update_ratings()
        return recommender_route(user_id, self.engine.collab_r)

    def hybrid_recommender_route(self, user_id: int):
        self.try_update_ratings()
        return recommender_route(user_id, self.engine.hybrid_r)

    def popularity_recommender_route(self):
        if recommendation_count := request.args.get('topn') is None:
            recommendation_count = 10
        if verbose := request.args.get('verbose') is None:
            verbose = False

        filtered_shows = None
        if query := request.args.get('query'):
            regex_string = query.replace(',', '|')
            filtered_shows = filter_by_query('genre', lambda x: bool(
                re.search(regex_string, x, re.IGNORECASE)), self.engine.items)

        shows = filtered_shows if (filtered_shows is not None) else self.engine.items
        recs = self.engine.popularity_r.generate_recommendations(
            shows, recommendation_count, verbose)
        return jsonify({"recommendations": recs.to_dict('records')})