from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

from myconstants import *


def calculate_popularity_stats(df: DataFrame):
    overall_vote_average = df['rating'].mean()
    ninetieth_percentile_member_count = df['members'].quantile(0.90)
    return overall_vote_average, ninetieth_percentile_member_count


def filter_by_query(column, condition, df: DataFrame) -> DataFrame:
    return df.copy().loc[[condition(x) for x in df[f'{column}']]]


def calculate_tfidf_matrix(df: DataFrame):
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.003,
                                 max_df=0.5,
                                 max_features=5000,
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(
        df['name'] + "" + df['genre'] + "" + df['type'])
    tfidf_feature_names = vectorizer.get_feature_names()

    return (tfidf_feature_names, tfidf_matrix)


def weight_rating(column: Series, min_vote_count, overall_vote_average):
    vote_count = column['members']
    rating = column['rating']
    # IMDB weighting formula
    return (vote_count/(vote_count + min_vote_count) * rating + min_vote_count/(vote_count + min_vote_count) * overall_vote_average)


def chunk_dataframe(df, chunk_size):
    return [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]


def add_anime_info(anime_df: DataFrame, db):
    """
    Adds anime title image objects and synposes to database records
    """
    cur = db.cursor()
    print('start')
    for i, anime_name in enumerate(anime_df["name"]):
        print(i, anime_name)
        try:
            anime_data = requests.get(f'{KITSUANIME_APIBASE}anime?filter[text]="{anime_name}"').json()[
                'data'][0]['attributes']
        except (KeyError, IndexError) as e:
            print("anime requested could not be found")
        else:
            synopsis = str(anime_data["synopsis"]).replace(
                '"', '').replace("'", '')
            coverImage = anime_data["coverImage"]

            if (coverImage is None):
                print(anime_name)
                coverImage = anime_data["posterImage"]

            coverImageString = str(coverImage).replace(
                "'", '"').replace("None", "null")

            cur.execute(f"UPDATE anime \
                            SET synopsis='{synopsis}', \
                            \"titleImage\"=\'{coverImageString}\' \
                            WHERE name='{anime_name}'")
            db.commit()
    print('done')
