from src.utils import calculate_popularity_stats, weight_rating
import pandas as pd


class PopularityRecommender:

    def generate_recommendations(self, shows: pd.DataFrame, recommendation_count: int = 10, verbose: bool = False) -> pd.DataFrame:

        overall_vote_average, member_minimum = calculate_popularity_stats(shows)
        filtered_recs = shows[shows['members'] >= member_minimum]
        filtered_recs['weighted_score'] = filtered_recs.apply(
            func=lambda show: weight_rating(show, member_minimum, overall_vote_average), axis=1)
        top_shows = filtered_recs.sort_values(by='weighted_score',
                                              ascending=False)[
                    :min(recommendation_count, len(filtered_recs))]

        if not verbose:
            top_shows = top_shows[["anime_id", "weighted_score"]]

        return top_shows
