import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from IPython.display import Video


def get_more_specific_df(df, column, value):
    df = df[df[column] == value]
    return df


def track_player_through_play(df, player, plotting=False):
    player_in_play = get_more_specific_df(df, 'displayName', player)
    return player_in_play[['x', 'y']]


tracking_csv = pd.read_csv('csvs/simple_tracking.csv')
players = np.unique(tracking_csv['displayName'])
for player in players:
    locations = track_player_through_play(tracking_csv, player)



# plt.xlim([0, 120])
# plt.ylim([0, 53.3])
