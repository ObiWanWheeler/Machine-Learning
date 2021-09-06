from .collaborative_recommender import CollabRecommender
from .content_recommender import ContentRecommender
import pandas as pd


class HybridRecommender():

    '''
        @param weights: weighting of content to collab recommendation. Dictionary of form {'collab': x, 'content': y}
    '''
    def __init__(self, content_recommender: ContentRecommender, collaborative_recommender: CollabRecommender, item_column, weights: dict={'collab': 10.0, 'content': 1.0}) -> None:
        self.cont_r = content_recommender
        self.collab_r = collaborative_recommender
        self.weights = weights
        self.item_column = item_column

    def give_recommendations(self, user_id, items_df, topn=10, verbose=False, items_to_ignore=[]):
        # top content based recommendations
        cont_recs_df = self.cont_r.give_recommendations(
            user_id, items_df, topn=1000, items_to_ignore=items_to_ignore).rename(columns={'relevance': 'relevance-Content'})

        # top collab based recommendatoins
        collab_recs_df = self.collab_r.give_recommendations(
            user_id, items_df, topn=1000, items_to_ignore=items_to_ignore).rename(columns={'relevance': 'relevance-Collab'})

        # merge results into one frame
        joint_recs_df = pd.merge(left=cont_recs_df, right=collab_recs_df,
                                 how='outer', left_on=self.item_column, right_on=self.item_column).fillna(0.0)
        joint_recs_df['joint_relevance_score'] = (joint_recs_df['relevance-Content'] * self.weights['content']) + (joint_recs_df['relevance-Collab'] * self.weights['collab'])

        joint_recs_sorted = joint_recs_df.sort_values(by='joint_relevance_score', ascending=False).head(topn)

        if verbose:
            if items_df is not None:
                joint_recs_sorted = joint_recs_sorted.merge(items_df, how='left', left_on=self.item_column, right_on=self.item_column)
            else:
                raise Exception('items_df must not be none to use verbose mode')

        return joint_recs_sorted
