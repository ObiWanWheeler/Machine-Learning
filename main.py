if __name__ == 'main':
    print('main')

import pandas as pd
from data import anime, ratings

def learn_new_user_preferences_test():
    abrev_anime = anime[anime['anime_id'] <= 10]
    abrev_ratings = ratings[ratings['user_id'] <= 10]

    abrev_ratings.to_csv('./data/anime/ratings_abrev.csv', index=False)

    new_user_id = abrev_ratings['user_id'].max() + 1
    user_ratings = []
    for anime_id in abrev_anime['anime_id']:
        print(anime.iloc[anime_id][['name', 'genre', 'type', 'episodes']])
        user_rating = input("give a rating from 0 to 10, or -1 if you haven't seen it:\n")
        user_ratings.append({'user_id': new_user_id, 'anime_id': anime_id, 'rating': user_rating})

    abrev_ratings = abrev_ratings.append(user_ratings, ignore_index=True)
    abrev_ratings.to_csv('./data/anime/ratings_abrev2.csv', index=False)


learn_new_user_preferences_test()