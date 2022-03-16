import pandas as pd
import numpy as np


def get_more_specific_df(df, column, value):
    df = df[df[column] == value]
    # Returns a df where all values of a certain column are a certain value
    return df


csv = pd.read_csv('csvs/tracking2020.csv')
receive_plays = get_more_specific_df(csv, 'event', 'punt_received')
play_nos = np.unique(receive_plays['playId'])
for id in play_nos:
    df = get_more_specific_df(csv, 'playId', id)
    games = np.unique(df['gameId'])
    if np.size(games) == 1:
        df.to_csv('csvs\\Receiving_Plays\\play' + str(id) + 'game' + str(games[0]) + '.csv')
    else:
        for game in games:
            new_df = get_more_specific_df(df, 'gameId', game)
            if 'punt_received' in np.unique(new_df['event']):
                new_df.to_csv('csvs\\Receiving_Plays\\play' + str(id) + '-game' + str(game) + '.csv')
