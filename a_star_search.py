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


'''
returner_pos = [61.68, 29.08]

csv = pd.read_csv('Frame1.csv').loc[:,['x', 'y', 't']]

xs = csv['x']
ys = csv['y']
ts = csv['t']

returner_speed = 7
'''

# This function carries out A* search at each frame of defenders. There is still
# a lot to consider e.g. whether the returner position inputted is the position
# calculated from the last A* search or the returners actual position. There is
# also no eventuality where there are no possible lines, and there is no eventuality
# for when the returner has passed through all of the defenders.

def a_star_search(lines, ts, returner_pos, returner_speed=7, goal_line=0, plotting=False):
    # Initialise xs and ys
    xs = lines[:, 0]
    ys = lines[:, 1]
    # Split up xs, ys and ts into groups of 20 (from the lines)
    arrays = []
    for array in [xs, ys, ts]:
        new_array = np.array_split(array, len(array)/20)
        arrays.append([list(arr) for arr in new_array])
    xs, ys, ts = arrays
    possible_points = []
    # Find the point on each line with the highest expected arrival time
    for i in range(np.shape(xs)[0]):
        max_t_index = ts[i].index(np.max(ts[i]))
        longest_time_loc = [xs[i][max_t_index], ys[i][max_t_index]]
        # Find the distance to goal from the point (shorter the better)
        scaled_dist_to_goal = distance_to_goal(longest_time_loc)/120
        # Find the time to the point and only consider in the algorithm if the returner can get through
        time_to_point = dist_between_points(returner_pos, longest_time_loc)/returner_speed/(120/returner_speed)
        if time_to_point < ts[i][max_t_index]:
            possible_points.append(Point(longest_time_loc, 2*time_to_point, scaled_dist_to_goal))
    scores = []
    # Get the weighted scores (the weighting is a bit wrong)
    for point in possible_points:
        scores.append(point.f)

    index_of_min = scores.index(np.min(scores))
    # Find the point that the returner should aim for, and plot an arrow below if plotting==True
    point = possible_points[index_of_min].xy
    if plotting:
        point_time = possible_points[index_of_min].g/2/(returner_speed/120)

        plt.scatter(returner_pos[0], returner_pos[1], c='r')
        for i in range(np.shape(xs)[0]):
            plt.plot(xs[i], ys[i], 'b-')


        x_to_point = (point[0] - returner_pos[0])*(0.1/point_time)
        y_to_point = (point[1] - returner_pos[1])*(0.1/point_time)

        plt.arrow(returner_pos[0], returner_pos[1], x_to_point, y_to_point, head_width=1)
    return point







# plt.plot(xs, ys)
# plt.show()
