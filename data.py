import pandas as pd

# data sets
anime = pd.read_csv("./data/anime/anime.csv", low_memory=False)
ratings = pd.read_csv("./data/anime/rating.csv", low_memory=False)