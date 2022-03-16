import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class Player:
    def __init__(self, name, xs, ys, team, speed):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.team = team
        self.speed = speed

    def getxyloc(self, i):
        x = self.xs.iloc[i]
        y = self.ys.iloc[i]
        return [x, y]


def distance(loc1, loc2):
    return np.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)


def returner(csv, frame):
    df = csv[csv['frameId'] == frame]
    football = df[df['displayName'] == 'football']
    football_location = [football['x'].iloc[0], football['y'].iloc[0]]
    distances = []
    for player in np.unique(df['displayName']):
        if player == 'football':
            distances.append(100000)
        else:
            player_location = [df[df['displayName'] == player]['x'].iloc[0], df[df['displayName'] == player]['y'].iloc[0]]
            distance_to_ball = distance(football_location, player_location)
            distances.append(distance_to_ball)
    min_distance_index = distances.index(np.min(distances))
    return np.unique([df['displayName']])[min_distance_index]


def get_points_of_defenders(defenders, index):
    return [defender.getxyloc(index) for defender in defenders]



# Find defensive locations in each frame
# Need to find the team on the ball and the defensive team

# Identify the delaunay triangles within the defensive structure

# Draw the delaunay triangles frame by frame

fig, ax = plt.subplots()
csv = pd.read_csv('csvs/Receiving_Plays/play116-game2021010301.csv')
receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
punt_returner = returner(csv, receive_frame)
attacking_team = csv[csv['displayName'] == punt_returner]['team'].iloc[0]
attackers = []
defenders = []
for player in np.unique(csv['displayName']):
    player_csv = csv[csv['displayName'] == player][receive_frame:]
    size = np.shape(player_csv)[0]
    team = csv[csv['displayName'] == player]['team'].iloc[0]
    if team == attacking_team:
        attackers.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
    else:
        defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))

for frame in range(size):
    points_def = np.array(get_points_of_defenders(defenders, frame))
    points_off = np.array(get_points_of_defenders(attackers, frame))
    tri = Delaunay(points_def)
    plt.triplot(points_def[:,0], points_def[:,1], tri.simplices)
    plt.plot(points_def[:,0], points_def[:,1], 'o', c='r')
    plt.plot(points_off[:,0], points_off[:,1], 'o', c='b')
    plt.pause(0.05)
    ax.clear()
plt.xlim([0, 120])
plt.ylim([0, 53.3])
plt.show()
