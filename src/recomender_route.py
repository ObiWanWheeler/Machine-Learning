from flask import jsonify

from src.recommenders.recommender import Recommender
from src.utils import get_query_vars


def recommender_route(user_id: int, recommender: Recommender):
    if user_id not in recommender.ratings['user_id'].values:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), \
               "400 this user has not rated any shows yet, so no recommendations can be made."

    recommendation_count, verbose = get_query_vars()
    print("verbose? : ", verbose)
    items_to_ignore = list(
        recommender.ratings[recommender.ratings['user_id'] == user_id]['anime_id'])
    print("generating recommendations...")
    recs = recommender.generate_recommendations(user_id=user_id, recommendation_count=recommendation_count,
                                                verbose=verbose, items_to_ignore=items_to_ignore)
    print("generated")
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})


