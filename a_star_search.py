import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Point:
    def __init__(self, xy, g, h):
        self.xy = xy
        self.g = g
        self.h = h

    def f(self):
        return self.g + self.h


def distance_to_goal(point, goal_line=0):
    return abs(point[0] - goal_line)


def dist_between_points(point_1, point_2):
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1]))


# a* takes direction into account

# need node weights for each step (g) and also a weight of each node to the end goal (h) (f = g+h)

# distance of the path + how far they have to go from this

# all nodes start with a distance of infinity

# frame by frame, find the node to travel through with the longest defender arrival times
# and the shortest distance to goal.

# connect point with the lowest risk in one window to the next, find the optimal path to the end zone

# What is the input? Returners Distance and Points.

# Updates continuously in time


returner_pos = [61.68, 29.08]

csv = pd.read_csv('Frame1.csv').loc[:,['x', 'y', 't']]

xs = csv['x']
ys = csv['y']
ts = csv['t']

returner_speed = 7

def a_star_search(lines, ts, returner_pos, returner_speed=7, goal_line=0, plotting=False):
    xs = lines[:, 0]
    ys = lines[:, 1]
    arrays = []
    for array in [xs, ys, ts]:
        new_array = np.array_split(array, len(array)/20)
        arrays.append([list(arr) for arr in new_array])
    xs, ys, ts = arrays
    possible_points = []
    for i in range(np.shape(xs)[0]):
        max_t_index = ts[i].index(np.max(ts[i]))
        longest_time_loc = [xs[i][max_t_index], ys[i][max_t_index]]
        scaled_dist_to_goal = distance_to_goal(longest_time_loc)/120
        time_to_point = dist_between_points(returner_pos, longest_time_loc)/returner_speed/(120/7)
        if time_to_point < ts[i][max_t_index]:
            possible_points.append(Point(longest_time_loc, 2*time_to_point, scaled_dist_to_goal))
    scores = []
    for point in possible_points:
        print("loc:", point.xy, ", g:", point.g, ", h:", point.h)
        scores.append(point.g + point.h)

    index_of_min = scores.index(np.min(scores))

    point = possible_points[index_of_min].xy
    if plotting:
        point_time = possible_points[index_of_min].g/2/(7/120)

        plt.scatter(returner_pos[0], returner_pos[1], c='r')
        for i in range(np.shape(xs)[0]):
            plt.plot(xs[i], ys[i], 'b-')


        x_to_point = (point[0] - returner_pos[0])*(0.1/point_time)
        y_to_point = (point[1] - returner_pos[1])*(0.1/point_time)

        plt.arrow(returner_pos[0], returner_pos[1], x_to_point, y_to_point, head_width=1)
    return point







# plt.plot(xs, ys)
# plt.show()
