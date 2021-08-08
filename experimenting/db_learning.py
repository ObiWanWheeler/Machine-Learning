import pandas as pd
from psycopg2.extras import execute_values

def input_user_ratings(anime_df, id):
    user_ratings = []
    for anime_id in anime_df['anime_id']:
        print(anime_df[anime_df['anime_id'] == anime_id][['name', 'genre', 'type', 'episodes']], "\n")
        user_rating = input("give a rating from 0 to 10, or -1 if you haven't seen it:\n")
        user_ratings.append([id, anime_id, user_rating])

        if input('leave another review? y/n: ') == 'n':
            break

    return user_ratings


def learn_new_user_preferences_db(connection):
    with connection.cursor() as cur:
        cur.execute('SELECT MAX(user_id) FROM rating')
        max_user_id = cur.fetchone()[0]
        new_user_id = max_user_id + 1

        cur.execute('SELECT * FROM anime')
        anime = cur.fetchall()
        anime_df = pd.DataFrame(anime, columns=['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'])

        user_ratings = input_user_ratings(anime_df, new_user_id)

        execute_values(cur, "INSERT INTO rating (user_id, anime_id, rating) VALUES %s", user_ratings)
        print(f'ratings added, your user ID is {new_user_id}')
    connection.commit()
    return new_user_id

        

