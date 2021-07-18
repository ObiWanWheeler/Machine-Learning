from pandas.core.frame import DataFrame
from utils import calculate_popularity_stats, weight_rating

class PopularityRecommender:

    def give_recommendation(df, topn):
        overall_vote_average, member_minimum = calculate_popularity_stats(df)
        df = df.copy()[df['members'] >= member_minimum]

        df['weighted_score'] = df.apply(
            weight_rating, df, member_minimum, overall_vote_average, axis='columns')
        df.sort_values(by=['weighted_score'],
                            ascending=False, inplace=True, kind='mergesort')
        return df[['name', 'members',
                        'weighted_score', 'genre']].head(topn)
