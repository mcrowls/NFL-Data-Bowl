import numpy as np
import statistics


def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


# Feed Convex Hull points
def boundary_windows(hull_points, returner_pos_x):
    # Remove points passed by runner
    bad_ind = []
    for i in range(0, len(hull_points)):
        if hull_points[i][0] > returner_pos_x:
            bad_ind.append(i)
    points_after_returner = []
    for i in range(0, len(hull_points)):
        if i not in bad_ind:
            points_after_returner.append(hull_points[i])

    points_x = [item[0] for item in points_after_returner]
    points_y = [item[1] for item in points_after_returner]


    ind_top = sorted(range(len(points_x)), key=lambda i: points_x[i])[-2:]
    #print(points_x)
    #print(np.argpartition(points_x, -1))
    #print(ind_top)
    ind_left = np.argmax(points_y)
    ind_right = np.argmin(points_y)
    ind_bottom = np.argpartition(points_x, -1)[0]


    x_lim =  2/3*(points_x[ind_top[-1]]+points_x[ind_bottom])
    y_lim = statistics.median(points_y) #2/3*(points_y[ind_left]+points_y[ind_right])

    left_window = [points_after_returner[ind_left], points_after_returner[ind_bottom]]
    right_window = [points_after_returner[ind_right] ,points_after_returner[ind_bottom]]
    top_window = [points_after_returner[ind_top[0]], points_after_returner[ind_top[1]]]

    for i in range(0, len(points_x)):
        if points_x[i] > x_lim:
            top_window.append(points_after_returner[i])
        if points_y[i] > y_lim:
            left_window.append(points_after_returner[i])
        if points_y[i] < y_lim:
            right_window.append(points_after_returner[i])

    return top_window, right_window, left_window