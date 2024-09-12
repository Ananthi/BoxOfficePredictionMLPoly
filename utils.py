import pandas as pd
import numpy as np
import ast

MOVIE_DATA = "./data/tmdb_5000_movies.csv"
# TMP_DUMP = "./data/md.csv"

all_genre_list = []


def get_all_genres(movie_df):

    all_genre_list = []
    for genre_list in movie_df["genres"]:
        for g in genre_list:
            if g["name"] not in all_genre_list:
                all_genre_list.append(g["name"])
    return all_genre_list


def check_gtype(glist, g):

    for g_movie in glist:
        if g_movie["name"] == g:
            return 1
    return 0


def load_data():
    movie_df = pd.read_csv(MOVIE_DATA)
    movie_df = movie_df[movie_df.revenue >= 1000000]
    # print(movie_df.describe)
    movie_df["genres"] = movie_df["genres"].apply(ast.literal_eval)

    all_genre_list = get_all_genres(movie_df)
    for g in all_genre_list:
        g_name = "is_" + g
        movie_df[g_name] = movie_df["genres"].apply(check_gtype, g=g)

    #    movie_df.to_csv(TMP_DUMP)

    XY_df = movie_df.filter(like="is_")
    # del XY_df["is_Foreign"]
    # del XY_df["is_Western"]
    # del XY_df["is_Documentary"]
    # del XY_df["is_Music"]
    # del XY_df["is_War"]
    # del XY_df["is_History"]
    # del XY_df["is_Mystery"]

    XY_df["revenue"] = movie_df["revenue"] / 100000000
    XY = XY_df.to_numpy()
    Y = XY[:, -1]
    X = np.delete(XY, -1, axis=1)
    # XY_df.to_csv(TMP_DUMP)
    X_train = X[1:]
    Y_train = Y[1:]
    return (X_train, Y_train)
