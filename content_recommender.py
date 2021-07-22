from pandas.core.frame import DataFrame
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:

    def __init__(self, user_profiles, tfidf_matrix, item_ids) -> None:
        self.user_profiles = user_profiles
        self.tfidf_matrix = tfidf_matrix
        self.item_ids = item_ids

    def get_similar_items_ids(self, user_id, topn=10):
        cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.tfidf_matrix)
        most_similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        return sorted(
            [
                (self.item_ids[i], cosine_similarities[0, i])
                for i in most_similar_indices
            ],
            key=lambda x: -x[1],
        )

    def give_recommendation(self, user_id, item_id_column, items_df=None, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self.get_similar_items_ids(user_id, topn)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations = DataFrame(similar_items_filtered, columns=[item_id_column, 'relevance'])

        if verbose:
            if items_df is not None:
                recommendations = recommendations.merge(items_df, how='left', left_on=item_id_column, right_on=item_id_column)
            else:
                raise Exception('items_df must not be none to use verbose mode')
        
        return recommendations

