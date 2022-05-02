# Get each player location and speed
# Divide the pitch into a grid (60 x 40)
# Calculate for each pixel who gets there first
# Whoever gets there first has control
# Add uncertainty
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from frechetdist import frdist



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

def extract_one_game(game):
    home = {}
    away = {}
    balls = []

    players = game.sort_values(['frameId'], ascending=True).groupby('nflId')
    for id, dx in players:
        jerseyNumber = int(dx.jerseyNumber.iloc[0])
        if dx.team.iloc[0] == "home":
            home[jerseyNumber] = list(zip(dx.x.tolist(), dx.y.tolist()))
        elif dx.team.iloc[0] == "away":
            away[jerseyNumber] = list(zip(dx.x.tolist(), dx.y.tolist()))

    ball_df = game.sort_values(['frameId'], ascending=True)
    ball_df = ball_df[ball_df.team == "football"]
    balls = list(zip(ball_df.x.tolist(), ball_df.y.tolist()))
    return home, away, balls

def get_pixels(pitch_size, grid_size):
    xs = np.linspace(0, pitch_size[0], grid_size[0])
    ys = np.linspace(0, pitch_size[1], grid_size[1])
    x_spacing = xs[1] - xs[0]
    y_spacing = ys[1] - ys[0]
    pixels = []
    for x in xs[:-1]:
        row = []
        for y in ys[:-1]:
            row.append([x + (x_spacing/2), y + (y_spacing/2)])
        pixels.append(row)
    return pixels, x_spacing, y_spacing

def get_closest(point, points):
    # Want to add in speed
    distances = []
    for other_point in points:
        distances.append(np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2))
    return distances.index(np.min(distances))

def assign_pixel_values(frame, home, away, pixels):
    # print(np.shape(pixels))
    values = np.zeros(np.shape(pixels)[:-1])
    home = [home[player][frame] for player in home]
    away = [away[player][frame] for player in away]
    for row in pixels:
        row_index = pixels.index(row)
        for pixel in row:
            pixel_index = row.index(pixel)
            closest_home = get_closest(pixel, home)
            closest_away = get_closest(pixel, away)
            closest_index = get_closest(pixel, [home[closest_home], away[closest_away]])
            # home = 0, away = 1
            if closest_index == 0:
                values[row_index][pixel_index] = 0
            else:
                values[row_index][pixel_index] = 1
    return values

def get_neighbours(pixel, x_spacing, y_spacing):
    neighbours = []
    if pixel[1] + y_spacing < 53.3:
        # neighbours.append([pixel[0], pixel[1] + y_spacing])
        neighbours.append([pixel[0] - x_spacing, pixel[1] + y_spacing])
    if pixel[1] - y_spacing > 0:
        # neighbours.append([pixel[0], pixel[1] - y_spacing])
        neighbours.append([pixel[0] - x_spacing, pixel[1] - y_spacing])
    neighbours.append([pixel[0] - x_spacing, pixel[1]])
    return neighbours

def measure_distance_to_blue(pixel, pixels, vector, pixel_values):
    value = pixel_values[int(pixel[0]-0.5)][int(pixel[1]-0.5)]
    truth = True
    counter = 0
    while truth == True:
        next_pixel = [pixel[0] + vector[0], pixel[1] + vector[1]]
        indicies = [int(next_pixel[0] - 0.5), int(next_pixel[1] - 0.5)]
        # print(indicies)
        if 0 > indicies[0] or indicies[0] >= 120 or 0 > indicies[1] or indicies[1] >= 52:
            truth = False
        elif pixel_values[indicies[0]][indicies[1]] == value:
            pixel = next_pixel
            counter += 1
        else:
            truth = False
        # if pixel_values
    return counter

def choose_direction(pixel, pixels, pixel_values, sideways_direction):
    sizes = []
    if sideways_direction == 0:
        vectors = [[-1, 1], [-1, 0], [-1, -1]]
    else:
        vectors = [[-1, 1], [-1, 0], [-1, -1], sideways_direction]
    for array in vectors:
        sizes.append(measure_distance_to_blue(pixel, pixels, array, pixel_values))
    if np.sum(sizes) == 0:
        return [0, 0]
    else:
        index = sizes.index(np.max(sizes))
        return vectors[index]

def which_sideways_direction(start_pixel, pixel_values):
    vectors = [[0, -1], [0, 1]]
    leftover_pixels = pixel_values[:int(start_pixel[0])]
    pixel_value = pixel_values[int(start_pixel[0])][int(start_pixel[1])]
    array_left = []
    array_right = []
    for i in range(np.shape(leftover_pixels)[0]):
        array_left.append(leftover_pixels[i][:int(start_pixel[1])])
        array_right.append(leftover_pixels[i][int(start_pixel[1]):])
    if np.size(array_left) == 0:
        return [0, 1]
    elif np.size(array_right) == 0:
        return [0, -1]
    left_fraction = np.count_nonzero(array_left == pixel_value)/np.size(array_left)
    right_fraction = np.count_nonzero(array_right == pixel_value)/np.size(array_right)
    return vectors[[left_fraction, right_fraction].index(np.max([left_fraction, right_fraction]))]

def find_critical_sideways_point(start, points):
    truth = True
    i = 3
    while truth == True:
        # print(np.shape(points)[0])
        # print(-start)
        # print(-i)
        if i > np.shape(points)[0]:
            return i - 1
        if points[-start][0] == points[-i][0]:
            i += 1
        else:
            point_index = i
            truth = False
            return point_index + 1

def ch_search(ball_position, returner_value, pixels, pixel_values, x_spacing, y_spacing):
    x = round(ball_position[0]) - 0.5
    if round(ball_position[1]) > ball_position[1] or ball_position[1] > 52.5:
        y = round(ball_position[1]) - 0.5
    else:
        y = round(ball_position[1]) + 0.5
    if pixel_values[int(x)-2][int(y)] != returner_value:
        x = round(ball_position[0]) + 0.5
    starting_node = [x, y]
    sideways_direction = which_sideways_direction(starting_node, pixel_values)
    truth = True
    points = []
    counter = 0
    while truth == True:
        sideways = False
        points.append(starting_node)
        direction = choose_direction(starting_node, pixels, pixel_values, sideways_direction)
        if direction == [0, 0]:
            if np.shape(points)[0] > 3 and points[-2][0] == points[-3][0]:
                counter += 1
                index = find_critical_sideways_point(2, points)
                starting_node = points[-index+2]
                points = points[:-index+2]
                if counter < 2:
                    sideways_direction = [sideways_direction[0], -sideways_direction[1]]
                else:
                    sideways_direction = 0
                sideways == True
            else:
                truth = False
        if sideways == False:
            starting_node = [starting_node[0] + direction[0], starting_node[1] + direction[1]]
    return points

def yards_gained(returner):
    return abs(returner[-1][0] - returner[0][0])

def find_n_yard_point(array, n):
    point1 = array[-2]
    point2 = array[-1]
    if point2[0] - point1[0] == 0:
        if array[0][0] > array[-1][0]:
            return [array[0][0] - n, array[-1][1]]
        else:
            return [array[0][0] + n, array[-1][0]]
    gradient = (point2[1] - point1[1])/(point2[0] - point1[0])
    intercept = point2[1] - gradient*point2[0]
    if array[0][0] > array[-1][0]:
        return [array[0][0] - n, gradient*(array[0][0] - n) + intercept]
    else:
        return [array[0][0] + n, gradient*(array[0][0] + n) + intercept]

def add_points_til_n(array, n):
    if len(array) == 1:
        return array
    starting_point = array[0]
    relevant_points = [array[0]]
    i = 1
    truth = False
    while truth == False:
        if abs(array[i][0] - starting_point[0]) < n:
            relevant_points.append(array[i])
        else:
            relevant_points.append(array[i])
            truth = True
        i += 1
        #If the path is less than 5 yards in total, need to stop
        if i == len(array):
            truth = True
    return relevant_points

def path_interpolate(array, n):
    points = []
    # print(n/len(array))
    for i in range(len(array)-1):
        xs = np.linspace(array[i][0], array[i+1][0], int(n/(len(array)-1)))
        ys = np.linspace(array[i][1], array[i+1][1], int(n/(len(array)-1)))
        for j in range(len(xs)):
            points.append([xs[j], ys[j]])
    return points

def remove_elements(array):
    new_array = []
    spacing = int(len(array)/10)
    for i in range(spacing):
        new_array.append(array[i*10])
    return new_array


def calc_frechet_distance(actual_path, predicted_path, n):
    # print(actual_path[-1])
    actual_path = add_points_til_n(actual_path, n)
    predicted_path = add_points_til_n(predicted_path, n)
    # print(predicted_path)
    #actual_path[-1] = find_five_point(actual_path)
    # predicted_path[-1] = find_n_yard_point(predicted_path, n)
    # actual_path[-1] = find_n_yard_point(actual_path, n)
    # print()
    num = np.lcm(len(predicted_path)-1, len(actual_path)-1)
    predicted_path = path_interpolate(predicted_path, num)
    actual_path = path_interpolate(actual_path, num)
    if len(predicted_path) > 500:
        predicted_path = remove_elements(predicted_path)
        actual_path = remove_elements(actual_path)
    frdist_val = frdist(predicted_path, actual_path)
    return frdist_val


def fraction_of_pitch(returner, pixel_values, last_defender):
    if returner[0][0] >= 119.5:
        index1 = round(returner[0][0]) - 1
    else:
        index1 = round(returner[0][0])
    if returner[0][1] >= 52.5:
        index2 = round(returner[0][1]) - 1
    else:
        index2 = round(returner[0][1])
    ret_team_value = pixel_values[index1][index2]
    limited_pixels = pixel_values[last_defender:index1]
    count = np.count_nonzero(limited_pixels == ret_team_value)
    return count/np.size(limited_pixels), ret_team_value

def pitch_control(csv):
    initial_pos = []
    home, away, balls = extract_one_game(csv)
    for home_player, away_player in zip(home, away):
        h_pos = home[home_player][0][0]
        a_pos = away[away_player][0][0]
        initial_pos.append(h_pos)
        initial_pos.append(a_pos)
    last_def = int(np.min(initial_pos))
    receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
    punt_returner = returner(csv, receive_frame)
    returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
    returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))

    pixels, x_spacing, y_spacing = get_pixels([120, 53], [121, 54])
    frame = receive_frame
    pixel_values = assign_pixel_values(frame, home, away, pixels)
    pitch_fraction, value = fraction_of_pitch(returner_pos, pixel_values, last_def)
    yards = yards_gained(returner_pos)
    points = ch_search(balls[frame], value, pixels, pixel_values, x_spacing, y_spacing)
    predicted_yards = yards_gained(points)
    frechet = calc_frechet_distance(returner_pos, points, 5)
    return returner_pos, points, home, away, frame, pixels, pixel_values, frechet, yards, pitch_fraction, predicted_yards
    