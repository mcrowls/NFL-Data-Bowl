def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

# Feed Convex Hull points
def boundary_windows(p_x, p_y, returner_pos_x):
    # Remove points passed by runner
    bad_ind = []
    for i in range(0, len(p_x)):
        if p_x[i] > returner_pos_x:
            bad_ind.append(i)
    points_x = []
    points_y = []
    for i in range(0, len(p_x)):
        if i not in bad_ind:
            points_x.append(p_x[i])
            points_y.append(p_y[i])

    ind_top = np.argpartition(points_x, -1)[-1]
    ind_left = np.argpartition(points_y, -1)[-1]
    ind_right = np.argpartition(points_y, -1)[1]
    ind_bottom = np.argpartition(points_x, -1)[1]

    top_left_rect = [points_x[ind_bottom] + 2/3*(points_x[ind_top]), (points_y[ind_left]-points_y[ind_right])/2,
                      1/3*(points_x[ind_top]), (points_y[ind_left]-points_y[ind_right])/2]

    mid_right_rect = [points_x[ind_bottom] + 1/3*(points_x[ind_top] - points_x[ind_bottom]), points_y[ind_right],
                      1 / 3 * (points_x[ind_top] - points_x[ind_bottom]),
                      (points_y[ind_left] - points_y[ind_right]) / 2]

    mid_left_rect = [points_x[ind_bottom] + 1 / 3 * (points_x[ind_top]), (points_y[ind_left] - points_y[ind_right]) / 2,
                     1 / 3 * (points_x[ind_top]), (points_y[ind_left] - points_y[ind_right]) / 2]

    low_right_rect = [points_x[ind_bottom] + 1 / 3 * (points_x[ind_top] - points_x[ind_bottom]), points_y[ind_right],
                      1 / 3 * (points_x[ind_top] - points_x[ind_bottom]),
                      (points_y[ind_left] - points_y[ind_right]) / 2]

    low_left_rect = [points_x[ind_bottom] + 1 / 3 * (points_x[ind_top]), (points_y[ind_left] - points_y[ind_right]) / 2,
                     1 / 3 * (points_x[ind_top]), (points_y[ind_left] - points_y[ind_right]) / 2]

    left_window = []
    right_window = []
    top_window = []

    for i in range(0, len(points_x)):
        point = (points_x[i], points_y[i])
        if point[0] >= points_x[ind_bottom] + 2/3*(points_x[ind_top] - points_x[ind_bottom]):
            top_window.append(point)
            if pointInRect(point, top_left_rect):
                left_window.append(point)
            else:
                right_window.append(point)

        elif pointInRect(point, mid_right_rect) or pointInRect(point, low_right_rect):
            right_window.append(point)

        elif pointInRect(point, mid_left_rect) or pointInRect(point, low_left_rect):
            left_window.append(point)

    return top_window, right_window, left_window
#
# # Get data before for smooth animation
#     for frame in range(size):
#         points_def.append(np.array(get_points_of_defenders(defenders, frame)))
#         points_off.append(np.array(get_points_of_defenders(attackers, frame)))
#         if delaunay:
#             bounds = ConvexHull(points_def[frame]).vertices
#
#             for element in bounds:
#                 bound_points_x.append(points_def[frame][element][0])
#                 bound_points_y.append(points_def[frame][element][1])
#
#             top_window, right_window, left_window = boundary_windows(bound_points_x, bound_points_y, returner_pos[frame][0])
#             all_top_windows.append(top_window)
#             all_left_windows.append(left_window)
#             all_right_windows.append(right_window)
#             outer_layer_x.append(bound_points_x)
#             outer_layer_y.append(bound_points_y)
#             bound_points_x = []
#             bound_points_y = []
#
#         tri = Delaunay(points_def[frame])