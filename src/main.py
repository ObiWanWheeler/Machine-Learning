import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import re
import time
import logging
from typing import Type
from data import get_connection_psycopg, DatabaseCustomORM
from recommenders.collaborative_recommender import CollabRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.popularity_recommender import PopularityRecommender
from recommenders.recommender import Recommender
from utils import filter_by_query

logging_format = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logging_format)
# creating FApp and connecting to database
app = Flask(__name__)

connection = get_connection_psycopg("./database.ini")
cursor = connection.cursor()

db = DatabaseCustomORM(cursor)


@app.route('/')
def index():
    return "Welcome to the film recommender API"


def fetch_feedback_data():
    ratings = db.fetch_all("rating")
    ratings_df = pd.DataFrame(
        ratings, columns=['user_id', 'anime_id', 'rating', 'createdAt', 'updatedAt'])
    ratings_df['rating'] = ratings_df['rating'].astype(np.int8)
    watched_ratings = ratings_df[ratings_df['rating'] != 0]

    return ratings_df, watched_ratings


def fetch_anime_data():
    anime = db.fetch_all("anime")
    anime_df = pd.DataFrame(anime, columns=[
        'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'updatedAt', 'createdAt', 'synopsis',
        'titleImage'])
    anime_df = anime_df[~anime_df["genre"].str.contains("Hentai")]
    anime_df.dropna(inplace=True)

    return anime_df


def recommender_route(user_id, recommender_class: Type[Recommender]):
    anime_df = fetch_anime_data()
    ratings_df, watched_ratings = fetch_feedback_data()

    recommender = recommender_class(anime_df, watched_ratings)

    if user_id not in ratings_df['user_id']:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), 400
    if recommendation_count := request.args.get('topn') is None:
        recommendation_count = 10
    if verbose := request.args.get('verbose') is None:
        verbose = False

    items_to_ignore = list(
        watched_ratings[watched_ratings['user_id'] == user_id]['anime_id'])

    recs = recommender.generate_recommendations(user_id=user_id, recommendation_count=recommendation_count,
                                                verbose=verbose, items_to_ignore=items_to_ignore)
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})


@app.route('/popularity-recommender')
def popularity_recommender():
    if recommendation_count := request.args.get('topn') is None:
        recommendation_count = 10
    if verbose := request.args.get('verbose') is None:
        verbose = False

    anime_df = fetch_anime_data()

    filtered_shows = None
    if query := request.args.get('query'):
        regex_string = query.replace(',', '|')
        filtered_shows = filter_by_query('genre', lambda x: bool(
            re.search(regex_string, x, re.IGNORECASE)), anime_df)

    shows = filtered_shows if (filtered_shows is not None) else anime_df

    recs = PopularityRecommender().generate_recommendations(
        shows, recommendation_count, verbose)
    return jsonify({"recommendations": recs.to_dict('records')})


@app.route('/content-recommender/<int:user_id>')
def content_recommender(user_id: int):
    return recommender_route(user_id, ContentRecommender)


@app.route('/collab-recommender/<int:user_id>')
def collab_recommender(user_id: int):
    return recommender_route(user_id, CollabRecommender)


@app.route('/hybrid-recommender/<int:user_id>')
def hybrid_recommender(user_id: int):
    return recommender_route(user_id, HybridRecommender)


@app.route('/user-added', methods=['POST'])
def user_added():
    pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)

cursor.close()
connection.close()
