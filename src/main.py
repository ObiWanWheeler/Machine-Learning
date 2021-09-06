import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

import data
from recommenders.collaborative_recommender import CollabRecommender
from recommenders.content_recommender import ContentRecommender
from recommenders.hybrid_recommender import HybridRecommender
from recommenders.profile_builder import ProfileBuilder
from utils import calculate_tfidf_matrix
from myconstants import *

# creating FApp and connecting to database
app = Flask(__name__)

connection = data.get_connection()
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
                        'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'updatedAt', 'createdAt'])

anime_ids = anime_df['anime_id'].to_list()


# initialising recommenders
tfidf_feature_names, tfidf_matrix = calculate_tfidf_matrix(anime_df)

profile_builder = ProfileBuilder(anime_df, anime_ids, 'user_id', 'anime_id',
                    'rating', watched_ratings, tfidf_matrix)
user_profiles = profile_builder.build_all_user_profiles()

collab_r = CollabRecommender(
    watched_ratings, user_id_column='user_id', item_id_column='anime_id', rating_column='rating')

content_r = ContentRecommender(user_profiles, tfidf_matrix, anime_ids, 'anime_id')

hybrid_r = HybridRecommender(content_r, collab_r, 'anime_id', {
    'content': 1.0, 'collab': DEFAULT_COLLAB_WEIGHT})


@app.route('/')
def index():
    return "Welcome to the film recommender API"


def recommender_route(user_id, recommendation_func):
    if user_id not in ratings_df['user_id']:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), 400
    topn = request.args.get('topn')
    if topn is None:
        topn = 10
    verbose = request.args.get('verbose')
    if verbose is None:
        verbose = False

    items_to_ignore = list(
            watched_ratings[watched_ratings['user_id'] == user_id]['anime_id'])

    recs = recommendation_func(user_id=user_id, items_df=anime_df, items_to_ignore=items_to_ignore, topn=topn, verbose=verbose)
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})

    
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
