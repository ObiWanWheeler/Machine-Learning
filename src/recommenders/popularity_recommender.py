from pandas.core.frame import DataFrame
from utils import calculate_popularity_stats, weight_rating

class PopularityRecommender:

    def give_recommendation(self, df, topn, verbose=False):
        overall_vote_average, member_minimum = calculate_popularity_stats(df)
        filtered_df = df.copy()[df['members'] >= member_minimum]

        filtered_df['weighted_score'] = filtered_df.apply(func=lambda column: weight_rating(column, member_minimum, overall_vote_average), axis='columns')
        filtered_df.sort_values(by=['weighted_score'],
                            ascending=False, inplace=True, kind='mergesort')
        if not verbose:
            return filtered_df[['anime_id', 'weighted_score']].head(topn)
        else:
            return filtered_df.head(topn)
