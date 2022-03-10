import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def get_more_specific_df(df, column, value):
    df = df[df[column] == value]
    return df


def track_player_through_play(df, player, plotting=False):
    player_in_play = get_more_specific_df(df, 'displayName', player)
    return player_in_play[['x', 'y']]


def home_or_away(df, player):
    player_df = get_more_specific_df(df, 'displayName', player)
    if player_df['team'][0] == 'home':
        return 'white'
    elif player_df['team'][0] == 'away':
        return 'blue'
    else:
        return 'brown'


tracking_csv = pd.read_csv('csvs/simple_tracking.csv')
# players = np.unique(tracking_csv['displayName'])
# fig = plt.figure(facecolor='green')
# ax = fig.gca()
# location_array = [track_player_through_play(tracking_csv, player) for player in players]
# num_frames = np.shape(location_array[-1])[0]
# camera=Camera(fig)
# for frame in range(num_frames):
#     for i in range(np.shape(location_array)[0]):
#         # colour = home_or_away(tracking_csv, players[i])
#         # print(location_array[i].iloc[frame]['x'])
#         ax.scatter(location_array[i].iloc[frame]['x'], location_array[i].iloc[frame]['y'], c='w')
#     # fig.xlim([0, 120])
#     # fig.ylim([0, 53.3])
#     camera.snap()
# animation = camera.animate()
# plt.close()
# animation.save('animation.gif', writer='PillowWriter', fps=10/1)
# Video('animation.gif')

fig = px.scatter(tracking_csv, x='x', y='y', hover_name='displayName', color='team', animation_frame='frameId', range_x=[0, 120], range_y=[0, 53.3])
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 50
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 20
fig.write_html('Animations/play.html')
fig.show()



    # plt.scatter(locations['x'], locations['y'])

    # plt.show()
