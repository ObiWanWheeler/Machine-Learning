from flask import jsonify, request

from src.recommenders.recommender import Recommender


def recommender_route(user_id: int, recommender: Recommender):
    if user_id not in recommender.ratings['user_id'].values:
        return jsonify({"error": "this user has not rated any shows yet, so no recommendations can be made."}), 400

    recommendation_count = request.args.get('topn')
    if recommendation_count is None:
        recommendation_count = 10
    else:
        recommendation_count = int(recommendation_count)

    verbose = request.args.get('verbose')
    if verbose is None:
        verbose = False
    else:
        verbose = verbose.lower() in ["true", "t", "yes", "y"]

    items_to_ignore = list(
        recommender.ratings[recommender.ratings['user_id'] == user_id]['anime_id'])

    recs = recommender.generate_recommendations(user_id=user_id, recommendation_count=recommendation_count,
                                                verbose=verbose, items_to_ignore=items_to_ignore)
    return jsonify({"user_id": user_id, "recommendations": recs.to_dict('records')})