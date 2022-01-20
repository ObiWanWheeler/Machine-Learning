import logging
import numpy as np
from flask import request
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse.linalg.eigen.arpack.arpack import svds
import requests
from sklearn.metrics import mean_squared_error

from myconstants import *
from src.exceptions import DimensionError


def calculate_popularity_stats(df: DataFrame):
    overall_vote_average = df['rating'].mean()
    ninetieth_percentile_member_count = df['members'].quantile(0.90)
    return overall_vote_average, ninetieth_percentile_member_count


def filter_by_query(column, condition, df: DataFrame) -> DataFrame:
    return df.copy().loc[[condition(x) for x in df[f'{column}']]]


def weight_rating(column: Series, min_vote_count, overall_vote_average):
    vote_count = column['members']
    rating = column['rating']
    # IMDB weighting formula
    return vote_count / (vote_count + min_vote_count) * rating + min_vote_count / (
            vote_count + min_vote_count) * overall_vote_average


def chunk_dataframe(df, chunk_size):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]


def dot_product(vec1, vec2):
    if len(vec1) != len(vec2):
        raise DimensionError("Vectors must have the same length to be multiplied")
    else:
        return sum(vec1[i] * vec2[i] for i in range(len(vec1)))


def add_anime_info(anime_df: DataFrame, db):
    """
    Adds anime title image objects and synopses to database records
    """
    cur = db.cursor()
    # add new columns to anime table
    cur.execute("ALTER TABLE anime ADD synopsis text")
    cur.execute("ALTER TABLE anime ADD titleImage jsonb")
    print('start')
    for i, anime_name in enumerate(anime_df["name"]):
        print(i, anime_name)
        # request the requested anime info from the kitsu anime API
        try:
            anime_data = requests.get(f'{KITSUANIME_APIBASE}anime?filter[text]="{anime_name}"').json()[
                'data'][0]['attributes']
        except (KeyError, IndexError) as e:
            print("anime requested could not be found")
        else:
            # filter through data to find synopsis and title image
            # format them in a way acceptable for postgres
            synopsis = str(anime_data["synopsis"]).replace(
                '"', '').replace("'", '')
            coverImage = anime_data["coverImage"]

            if coverImage is None:
                print(anime_name)
                coverImage = anime_data["posterImage"]

            coverImageString = str(coverImage).replace(
                "'", '"').replace("None", "null")
            # update relevant anime with synopsis and title image data
            cur.execute(f"UPDATE anime \
                            SET synopsis='{synopsis}', \
                            \"titleImage\"=\'{coverImageString}\' \
                            WHERE name='{anime_name}'")
            db.commit()
    print('done')


def find_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """finds the singular value decomposition of a matrix
    
    Parameters
    ----------
    matrix:
        the matrix to decompose
    Returns
    -------
    U:
        Left factor of matrix as np.ndarray
    V_T:
        Right factor of matrix as np.ndarray
    S:
        1D array of singular values of matrix
    """
    left_transpose = matrix.T @ matrix
    # calculate left singular vectors
    eigen_values, left_singular_vectors = np.linalg.eigh(left_transpose)
    right_transpose = matrix @ matrix.T
    # calculate right singular vectors
    _, right_singular_vectors = np.linalg.eigh(right_transpose)
    # calculate singular values
    logging.debug(eigen_values)
    singular_values = np.sqrt(eigen_values)

    return right_singular_vectors, singular_values, left_singular_vectors.T


def rank_reduce_matrix(matrix, k):
    """Calculates the best low rank factor approximation to a matrix using SVD
    
    Parameters
    ----------
    matrix:
        The matrix to rank reduce
    k:
        The rank of the approximated factors
    
    Returns
    -------
    Rank reduced factors of matrix as np.ndarrays
    """
    # find the singular value decomposition of the matrix
    left_singular_vectors, singular_values, right_singular_vectors_T = find_svd(matrix)
    # find_svd returns singular values in a 1d array, 
    singular_diagonal_matrix = np.zeros(matrix.shape)
    # so populate the proper sized array with them.
    np.fill_diagonal(singular_diagonal_matrix, singular_values)

    # rank reduce each matrix by taking the first k rows of S and the first k singular vectors of U and V_T
    left_rank_reduced = left_singular_vectors[:, :k]
    singular_vals_rank_reduced = singular_diagonal_matrix[:k, :k]
    right_T_rank_reduced = right_singular_vectors_T[:k, :]

    # calculate low rank factors
    left_factor = left_rank_reduced
    right_factor = singular_vals_rank_reduced @ right_T_rank_reduced

    return left_factor, right_factor


def rank_reduce_matrix_scipy(matrix, k):
    u, sigma, vt = svds(matrix, k=k)
    sigma = np.diag(sigma)
    return (u @ sigma).T, vt


def calc_mean_squared_error(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(actual, prediction)


def get_query_vars():
    recommendation_count = request.args.get('recommendationCount')
    if recommendation_count is None:
        recommendation_count = 10
    else:
        recommendation_count = int(recommendation_count)
    verbose = request.args.get('verbose')
    if verbose is None:
        verbose = False
    else:
        verbose = verbose.lower() in ["true", "t", "yes", "y"]
    return recommendation_count, verbose
