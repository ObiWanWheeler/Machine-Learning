import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import re

import data
from recommenders.collaborative_recommender import CollabRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.popularity_recommender import PopularityRecommender
from recommenders.profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix, filter_by_query, add_anime_info
from myconstants import *

# creating FApp and connecting to database
app = Flask(__name__)

connection = data.get_connection("./database.ini")
cursor = connection.cursor()


# fetching anime and rating data from database
cursor.execute("SELECT * FROM rating")
ratings = cursor.fetchall()
ratings_df = pd.DataFrame(
    ratings, columns=['user_id', 'anime_id', 'rating', 'createdAt', 'updatedAt'])
ratings_df['rating'] = ratings_df['rating'].astype(np.int8)
watched_ratings = ratings_df[ratings_df['rating'] != 0]

cursor.execute("SELECT * FROM anime")
anime = cursor.fetchall()
anime_df = pd.DataFrame(anime, columns=[
                        'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'updatedAt', 'createdAt', 'synopsis', 'titleImage'])


# initialising recommenders

content_r = ContentRecommender(anime_df, ratings_df)
print(content_r.generate_recommendations(1, 10, True))


@app.route('/')
def index():
    return "Welcome to the film recommender API"


def recommender_route(user_id, recommendation_func):
    if user_id not in ratings_df['user_id']:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), 400
    if topn := request.args.get('topn') is None:
        topn = 10
    if verbose := request.args.get('verbose') is None:
        verbose = False

    items_to_ignore = list(
            watched_ratings[watched_ratings['user_id'] == user_id]['anime_id'])

    recs = recommendation_func(user_id=user_id, items_df=anime_df, items_to_ignore=items_to_ignore, topn=topn, verbose=verbose)
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})


@app.route('/popularity-recommender')
def popularity_recommender():
    if query := request.args.get('query').split(','):
        regex_string = '|'.join(query)
        filtered_shows = filter_by_query('genre', lambda x: bool(re.search(regex_string, x, re.IGNORECASE)))
    

    
@app.route('/content-recommender/<int:user_id>')
def content_recommender(user_id: int):
    return recommender_route(user_id, content_r.give_recommendations)


@app.route('/collab-recommender/<int:user_id>')
def collab_recommender(user_id: int):
    return recommender_route(user_id, collab_r.give_recommendations)


@app.route('/hybrid-recommender/<int:user_id>')
def hybrid_recommender(user_id: int):
    return recommender_route(user_id, hybrid_r.give_recommendations)


@app.route('/user-added', methods=['POST'])
def user_added():
    pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)


cursor.close()
connection.close()
