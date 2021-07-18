from pandas.core.frame import DataFrame


def calculate_popularity_stats(df: DataFrame):
    overall_vote_average = df['rating'].mean()
    ninetieth_percentile_member_count = df['members'].quantile(0.90)
    return overall_vote_average, ninetieth_percentile_member_count


def filter_by_query(column, condition, df: DataFrame) -> DataFrame:
    return df.copy().loc[[condition(x) for x in df[f'{column}']]]


def weight_rating(df: DataFrame, min_vote_count, overall_vote_average):
    vote_count = df['members']
    rating = df['rating']
    # IMDB weighting formula
    return (vote_count/(vote_count + min_vote_count) * rating + min_vote_count/(vote_count + min_vote_count) * overall_vote_average)