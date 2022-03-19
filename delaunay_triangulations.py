from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skspatial.objects import Line
from skspatial.objects import Point
import math

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

def get_lines_from_delaunay(triangles,defenders):
    #Splitting the delaunay triangles into their sides
    index_pairs = []
    for tri in triangles.simplices:
        index_pairs.append(tri[0:2])
        index_pairs.append(tri[1:])
        index_pairs.append(tri[::2])

    #finding the defenders at the end of each line
    defender_pairs = []
    for pair in index_pairs:
        defender_pairs.append([defenders[pair[0]],defenders[pair[1]]])

    #finding the equally spaced points between each defender pair
    points = []
    for pair in defender_pairs:
        #21 is arbitrary, but then the head of the list is removed because it's on top of a defender 
        points.append(np.linspace(pair[0],pair[1],21,endpoint=False)[1:])
    points = np.array(points)
    return np.reshape(points,(-1,2))

#calculating the arrival time of each defender to each point in the window, but only keeping the min time
#These comments came from the competition code, detailing how to calculate the time penalty each blocker imposes on a defender
  #   1. Create a straight line between the defender and the target location
  #   2. Find the perpendicular projection of each blocker onto the line from Step 1
  #   3. If this projection does not lie in between the defender and the target, then penalty = 0
  #      Else use a Gaussian kernel with StdDev = the distance between the defender and the blocker's 
  #      perpendicular projection and x = the blocker's distance away from the perpendicular projection 
  #      to obtain the time penalty for the defender based on the blocker's position (multiplied by the)
  #      max_time_penalty parameter
def get_arrival_times(points,defenders, blockers):
    times = []
    for p in points:
        min_dist = float('inf')
        for d in defenders:
            dist = np.linalg.norm(d-p)
            if dist < min_dist:
                min_dist = dist
        total_penalty = 0
        for b in blockers:
            #path connecting defender to target point
            line = Line.from_points(Point(d),Point(p))
            #projected point of blocker onto path
            projected_point = line.project_point(b)
            defender_to_projection = np.linalg.norm(d-projected_point)
            projection_to_target = min_dist - defender_to_projection
            blocker_to_projection = np.linalg.norm(b-projected_point)
            #if blocker projection is not between defender and target, penalty is 0
            #Gaussian kernel uses a weighting term of 5, not sure why, the code used it before with no explanation
            penalty = 0 if (defender_to_projection >= min_dist or projection_to_target >= min_dist) else 5 * (1 / ((defender_to_projection) * math.sqrt(2*math.pi))) * math.exp(-(1/2) * (blocker_to_projection/(defender_to_projection))**2)
            total_penalty += penalty
        # ! This is the speed which needs to be variable later !
        time = min_dist / 7 + total_penalty
        times.append(time)
    return times


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
    arrival_points = None
    points_off = np.array(get_points_of_defenders(attackers, frame))
    tri = Delaunay(points_def)
    lines = get_lines_from_delaunay(tri,points_def)
    times = get_arrival_times(lines,points_def,points_off)

    plt.triplot(points_def[:,0], points_def[:,1], tri.simplices)
    plt.plot(points_def[:,0], points_def[:,1], 'o', c='r',label='Defenders')
    plt.plot(points_off[:,0], points_off[:,1], 'o', c='b', label='Attackers')
    p = plt.scatter(lines[:,0],lines[:,1],c=times, cmap = "RdYlGn",marker="s",s=5)
    cbar = fig.colorbar(p)
    cbar.set_label("Expected defender arrival time (s)")
    plt.legend(loc='best')
    plt.xlim([0, 120])
    plt.ylim([0, 53.3])
    plt.xlabel('x')
    plt.ylabel('y')
    if frame < size-1:
        plt.pause(0.05)
        ax.clear()
        fig.clear()
plt.show()
