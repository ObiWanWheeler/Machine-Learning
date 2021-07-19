from pandas.core.frame import DataFrame
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:

    def get_similar_items_ids(self, user_profile, tfidf_matrix, item_ids, topn=10):
        cosine_similarities = cosine_similarity(user_profile, tfidf_matrix)
        most_similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        return sorted(
            [
                (item_ids[i], cosine_similarities[0, i])
                for i in most_similar_indices
            ],
            key=lambda x: -x[1],
        )

    def give_recommendation(self, user_profile, tfidf_matrix, item_ids, items_df=None, items_to_ignore=[], topn=10):
        similar_items = self.get_similar_items_ids(user_profile, tfidf_matrix, item_ids, topn)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations = DataFrame(similar_items_filtered, columns=['anime_id', 'relevance'])

        if items_df is not None:
            recommendations = recommendations.merge(items_df, how='left', left_on='anime_id', right_on='anime_id')
        
        return recommendations

