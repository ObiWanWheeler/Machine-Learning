import csv_processes


def input_user_ratings(anime, id):
    user_ratings = []
    for anime_id in anime['anime_id']:
        print(anime[anime['anime_id'] == anime_id][['name', 'genre', 'type', 'episodes']])
        user_rating = input("give a rating from 0 to 10, or -1 if you haven't seen it:\n")
        user_ratings.append({'user_id': id, 'anime_id': anime_id, 'rating': user_rating})

        if input('leave another review? y/n: ') == 'n':
            break

    return user_ratings

# used for data in csv format, which is no longer how things go
@DeprecationWarning
def learn_new_user_preferences_abreviated_csv():
    
    abrev_anime = csv_processes.anime[csv_processes.anime['anime_id'] <= 10]
    abrev_ratings = csv_processes.feedback[csv_processes.feedback['user_id'] <= 10]

    abrev_ratings.to_csv('./data/anime/ratings_abrev.csv', index=False)

    new_user_id = abrev_ratings['user_id'].max() + 1
    user_ratings = input_user_ratings(abrev_anime, new_user_id)

    abrev_ratings = abrev_ratings.append(user_ratings, ignore_index=True)
    abrev_ratings.to_csv('./data/anime/ratings_abrev2.csv', index=False)

    print(f'your user_id is {new_user_id}')


@DeprecationWarning
def learn_new_user_preferences_csv():
    new_user_id = csv_processes.feedback['user_id'].max() + 1
    user_ratings = input_user_ratings(csv_processes.anime, new_user_id)
        
    ratings = csv_processes.feedback.append(user_ratings, ignore_index=True)
    ratings = ratings.to_csv('./data/anime/rating.csv', index=False)

    print(f'your user_id is {new_user_id}')