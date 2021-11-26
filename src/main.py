import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import re
import time
import logging
import data
from recommenders.collaborative_recommender import CollabRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.popularity_recommender import PopularityRecommender
from recommenders.profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix, filter_by_query, add_anime_info, find_svd
from myconstants import *

logging_format = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logging_format)
# creating FApp and connecting to database
app = Flask(__name__)

connection = data.get_connection("./database.ini")
cursor = connection.cursor()

# fetching anime and rating data from database
cursor.execute('SELECT * FROM rating')
ratings = cursor.fetchall()
ratings_df = pd.DataFrame(
    ratings, columns=['user_id', 'anime_id', 'rating', 'createdAt', 'updatedAt'])
ratings_df['rating'] = ratings_df['rating'].astype(np.int8)
watched_ratings = ratings_df[ratings_df['rating'] != 0]

cursor.execute("SELECT * FROM anime")
anime = cursor.fetchall()
anime_df = pd.DataFrame(anime, columns=[
                        'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'updatedAt', 'createdAt', 'synopsis', 'titleImage'])
anime_df = anime_df[~anime_df["genre"].str.contains("Hentai")]
anime_df.dropna(inplace=True)

# initialising recommenders
popularity_r = PopularityRecommender()
print(popularity_r.generate_recommendations(anime_df, 10, True))

start = time.perf_counter()
content_r = ContentRecommender(anime_df, watched_ratings)
print(content_r.generate_recommendations(1, 10, True)
      [["name", "genre", "relevance_score"]])
print(content_r.user_embeddings[1])
end = time.perf_counter()
logging.debug(f"Content recommendation took {end - start} to execute")


start = time.perf_counter()
collab_r = CollabRecommender(anime_df, watched_ratings)
print(collab_r.generate_recommendations(1, 10, True)
      [['name', 'genre', "predicted_rating"]])
end = time.perf_counter()
logging.debug(f"Collab recommendation took {end - start} to execute")


hybrid_r = HybridRecommender(anime_df, watched_ratings, content_r, collab_r)
print(hybrid_r.generate_recommendations(1, 10, True)[
      ['name', 'genre', 'relevance_score', 'predicted_rating', 'joint_relevance_score']])


@app.route('/')
def index():
    return "Welcome to the film recommender API"


def recommender_route(user_id, recommendation_func):
    if user_id not in ratings_df['user_id']:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), 400
    if recommendation_count := request.args.get('topn') is None:
        recommendation_count = 10
    if verbose := request.args.get('verbose') is None:
        verbose = False

    items_to_ignore = list(
        watched_ratings[watched_ratings['user_id'] == user_id]['anime_id'])

    recs = recommendation_func(user_id=user_id, recommendation_count=recommendation_count,
                               verbose=verbose, items_to_ignore=items_to_ignore)
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})


@app.route('/popularity-recommender')
def popularity_recommender():
    if recommendation_count := request.args.get('topn') is None:
        recommendation_count = 10
    if verbose := request.args.get('verbose') is None:
        verbose = False

    filtered_shows = None
    if query := request.args.get('query'):
        regex_string = query.replace(',', '|')
        filtered_shows = filter_by_query('genre', lambda x: bool(
            re.search(regex_string, x, re.IGNORECASE)), anime_df)

    shows = filtered_shows if (filtered_shows is not None) else anime_df

    recs = popularity_r.generate_recommendations(
        shows, recommendation_count, verbose)
    return jsonify({"recommendations": recs.to_dict('records')})


@app.route('/content-recommender/<int:user_id>')
def content_recommender(user_id: int):
    return recommender_route(user_id, content_r.generate_recommendations)


@app.route('/collab-recommender/<int:user_id>')
def collab_recommender(user_id: int):
    return recommender_route(user_id, collab_r.generate_recommendations)


@app.route('/hybrid-recommender/<int:user_id>')
def hybrid_recommender(user_id: int):
    return recommender_route(user_id, hybrid_r.generate_recommendations)


@app.route('/user-added', methods=['POST'])
def user_added():
    pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)


cursor.close()
connection.close()
