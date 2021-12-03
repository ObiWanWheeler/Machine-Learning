import functools
import operator

import numpy as np
import pandas as pd

from src.myconstants import PROD
from src.utils import calc_mean_squared_error


def calc_validity_stats(predictions, actual_data):
    """
    Calculates various error statistics for a prediction
    Parameters
    ------------
    predictions:
        prediction matrix to test for error
    actual_data:
        the original matrix of values
    Returns
    ---------
    tuple containing all the calculated statistics:
        root mean squared error, error stat 2, error stat 3, ...
    """
    return calc_mean_squared_error(predictions[actual_data.nonzero()], actual_data[actual_data.nonzero()])


def on_epoch_start(epoch_num):
    """
    Development oriented processes to happen at the start of each epoch.
    Mostly logging.
    """
    print(f"Epoch: {epoch_num + 1}")


def calc_sgd_predictions(feedback_df: pd.DataFrame, *, max_epoch_count=1000, latent_feature_count=75, alpha=0.01,
                         gamma=0.4,
                         accepted_deviation=2.5) -> pd.DataFrame:
    """Calculates prediction matrix by using matrix factorisation and stochastic gradient descent

    :param feedback_df: pd.Dataframe the sparse user / item matrix to predict
    :param max_epoch_count : int > 0 max number of times to iterate gradient descent to improve prediction accuracy
    :param latent_feature_count : int Number of latent factors to discover - rank to reduce data matrix to.
        The lower this is the faster to execute
    :param alpha: int
        'learning rate'. Size of steps to take in gradient descent.
        The lower this is the slower to execute
        but the more accurate results that can eventually be obtained.
    :param gamma: int
        regularization parameter, should be in the range 0.1 - 1.0. Higher values restrict
        model further.
    :param accepted_deviation:
        the point at which the approximation is 'good enough'. Lower this is the better,
        but the longer it will take.
    :return populated pivoted dataframe of user id against anime id with predicted ratings as values.
    """

    if max_epoch_count <= 0:
        max_epoch_count = 1000

    # work with numpy arrays instead of dfs
    # demean data to help mitigate any user bias
    feedback_matrix = feedback_df.fillna(0.0).to_numpy()

    m, n = feedback_matrix.shape
    user_lfs = np.random.rand(latent_feature_count, m)
    item_lfs = np.random.rand(latent_feature_count, n)

    # the user - rating pairs of ratings that have actually been left
    user, items = feedback_matrix.nonzero()

    predictions = []

    for epoch in range(max_epoch_count):  # iterate gradient descent
        if not PROD:
            on_epoch_start(epoch)  # development logging stuff
        predictions = user_lfs.T @ item_lfs  # make an approximation
        # check how good the approximation is
        mse = calc_validity_stats(predictions, feedback_matrix)
        if not PROD:
            print("MSE:", mse)

        # check if the approximation is 'good enough'
        if mse < accepted_deviation:
            print(f"broke after {epoch + 1} epochs")
            break

        # if not, iterate gradient descent again and improve our approximation
        for u, i in zip(user, items):
            dot = np.dot(user_lfs[:, u].T, item_lfs[:, i])
            difference = feedback_matrix[u, i] - dot
            user_lfs[:, u] += 2 * alpha * (difference * item_lfs[:, i] - gamma * user_lfs[:, u])
            item_lfs[:, i] += 2 * alpha * (difference * user_lfs[:, u] - gamma * item_lfs[:, i])

    return predictions


def calculate_similarity_score(user_embedding, show_embedding, genre_frequencies) -> int:
    """Calculates similarity between user embedding and show embedding,
    both given as dictionaries of form genre: score

    Keyword arguments:
    user_embedding -- dictionary of form genre: score for the user. Shows users preferences
    show_embedding -- dictionary of form genre: score for the show. Shows show's genres
    Return: return_description
    """

    # if len(user_embedding.keys()) != len(show_embedding.keys()):
    #     raise Exception("Cannot dot product vectors of different lengths")
    # as it turns out, due to the nature of the data, they may well have different sizes yet still be valid.
    # We need consider only the fields present on both.

    total_score = 0
    for genre, score in show_embedding.items():
        try:
            normalized_score = score / (0.1 * genre_frequencies[genre])
            total_score += user_embedding[genre] * normalized_score
        except KeyError:  # user didn't have this genre in their embedding, unlikely but plausible
            pass
    # penalty that prevents shows with loads of genres being recommended too highly
    total_score /= (len(user_embedding) / len(show_embedding))
    return total_score


def calculate_term_frequencies(items: pd.DataFrame, field: str) -> dict:
    # vectorized approach to iterating over HUGE array.
    terms_non_distinct = functools.reduce(operator.concat, items[field].apply(lambda row: row.split(', ')))
    terms_distinct = set(terms_non_distinct)
    return {term: terms_non_distinct.count(term) for term in terms_distinct}


def calculate_item_embeddings(items: pd.DataFrame) -> dict:
    """Calculates ALL item embeddings for item dataframe passed to self. These should never need to be changed.
    """

    embeddings: dict = {}

    show_df_cols = list(items.columns)
    show_tuple_anime_id_index = show_df_cols.index("anime_id")  # index of id column once df is numpy-ified
    show_tuple_genre_index = show_df_cols.index("genre")  # as above but for genre column

    shows_arr = np.array(items)  # convert df to numpy array, as they are far quicker to work with
    for show in shows_arr:  # iterate through each show
        embeddings[show[show_tuple_anime_id_index]] = {}  # initialise this shows embedding
        this_shows_genres = show[show_tuple_genre_index].split(", ")  # extract the show's genres
        for genre in this_shows_genres:  # construct embedding based on genres
            if genre not in embeddings[show[show_tuple_anime_id_index]]:
                embeddings[show[show_tuple_anime_id_index]][genre] = 1  # if a genre is present it gets a 1
            else:  # fairly sure this never gets entered, but just in case ;)
                embeddings[show[show_tuple_anime_id_index]][genre] += 1
    return embeddings
