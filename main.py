import data


def learn_new_user_preferences_test(test: bool):
    
    abrev_anime = data.anime[data.anime['anime_id'] <= 10]
    abrev_ratings = data.ratings[data.ratings['user_id'] <= 10]

    abrev_ratings.to_csv('./data/anime/ratings_abrev.csv', index=False)

    new_user_id = abrev_ratings['user_id'].max() + 1
    user_ratings = []
    for anime_id in abrev_anime['anime_id']:
        print(abrev_anime.iloc[anime_id][['name', 'genre', 'type', 'episodes']])
        user_rating = input("give a rating from 0 to 10, or -1 if you haven't seen it:\n")
        user_ratings.append({'user_id': new_user_id, 'anime_id': anime_id, 'rating': user_rating})

    abrev_ratings = abrev_ratings.append(user_ratings, ignore_index=True)
    abrev_ratings.to_csv('./data/anime/ratings_abrev2.csv', index=False)


def learn_new_user_preferences():
    new_user_id = data.ratings['user_id'].max() + 1
    user_ratings = []
    for anime_id in data.anime['anime_id']:
        print(data.anime.iloc[anime_id][['name', 'genre', 'type', 'episodes']])
        user_rating = input("give a rating from 0 to 10, or -1 if you haven't seen it:\n")
        user_ratings.append({'user_id': new_user_id, 'anime_id': anime_id, 'rating': user_rating})
        
        if input('leave another review? y/n: ') == 'y':
            break

    ratings = data.ratings.append(user_ratings, ignore_index=True)
    ratings = ratings.to_csv('./data/anime/rating.csv', index=False)

    print(f'your user_id is {new_user_id}')


if __name__ == 'main':
    connection = data.get_connection()

    connection.close()