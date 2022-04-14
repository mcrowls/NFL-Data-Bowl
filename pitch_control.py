# Get each player location and speed
# Divide the pitch into a grid (60 x 40)
# Calculate for each pixel who gets there first
# Whoever gets there first has control
# Add uncertainty
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def drawPitch(width, height, color="w"):
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, width + 30), ylim=(-15, height + 5))
    plt.axis('off')

    # Grass around pitch
    rect = patches.Rectangle((-10, -5), width + 40, height + 10, linewidth=1, facecolor='#3f995b', capstyle='round')
    ax.add_patch(rect)
    ###################

    # Pitch boundaries
    rect = plt.Rectangle((0, 0), width + 20, height, ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    ###################
    # vertical lines - every 5 yards
    for i in range(21):
        plt.plot([10 + 5 * i, 10 + 5 * i], [0, height], c="w", lw=2)
    ###################

    # distance markers - every 10 yards
    for yards in range(10, width, 10):
        yards_text = yards if yards <= width / 2 else width - yards
        # top markers
        plt.text(10 + yards - 2, height - 7.5, yards_text, size=20, c="w", weight="bold")
        # botoom markers
        plt.text(10 + yards - 2, 7.5, yards_text, size=20, c="w", weight="bold", rotation=180)
    ###################
        # yards markers - every yard
        # bottom markers
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [1, 3], color="w", lw=3)

        # top markers
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [height - 1, height - 3], color="w", lw=3)

        # middle bottom markers
        y = (height - 18.5) / 2
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [y, y + 2], color="w", lw=3)
    # middle top markers
    for x in range(20):
        for j in range(1, 5):
            plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [height - y, height - y - 2], color="w", lw=3)
    ###################

    # draw home end zone
    plt.text(2.5, (height - 10) / 2, "HOME", size=40, c="w", weight="bold", rotation=90)
    rect = plt.Rectangle((0, 0), 10, height, ec=color, fc="#0064dc", lw=2)
    ax.add_patch(rect)

    # draw away end zone
    plt.text(112.5, (height - 10) / 2, "AWAY", size=40, c="w", weight="bold", rotation=-90)
    rect = plt.Rectangle((width + 10, 0), 10, height, ec=color, fc="#c80014", lw=2)
    ax.add_patch(rect)
    ###################

    # draw extra spot point
    # left
    y = (height - 3) / 2
    plt.plot([10 + 2, 10 + 2], [y, y + 3], c="w", lw=2)

    # right
    plt.plot([width + 10 - 2, width + 10 - 2], [y, y + 3], c="w", lw=2)
    ###################

    # draw goalpost
    goal_width = 6  # yards
    y = (height - goal_width) / 2
    # left
    plt.plot([0, 0], [y, y + goal_width], "-", c="y", lw=10, ms=20)
    # right
    plt.plot([width + 20, width + 20], [y, y + goal_width], "-", c="y", lw=10, ms=20)

    return fig, ax


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
        if pixel_values[indicies[0]][indicies[1]] == value:
            pixel = next_pixel
            counter += 1
        else:
            truth = False
        # if pixel_values
    return counter


def choose_direction(pixel, pixels, pixel_values):
    sizes = []
    vectors = [[-1, 1], [-1, 0], [-1, -1]]
    for array in vectors:
        sizes.append(measure_distance_to_blue(pixel, pixels, array, pixel_values))
    if np.sum(sizes) == 0:
        return [0, 0]
    else:
        index = sizes.index(np.max(sizes))
        return vectors[index]



def a_star_search(ball_position, pixels, pixel_values, x_spacing, y_spacing):
    if round(ball_position[1]) > ball_position[1]:
        y = round(ball_position[1]) - 0.5
    else:
        y = round(ball_position[1]) + 0.5
    x = round(ball_position[0]) + 0.5
    starting_node = [x, y]
    truth = True
    points = []
    while truth == True:
        points.append(starting_node)
        direction = choose_direction(starting_node, pixels, pixel_values)
        if direction == [0, 0]:
            truth = False
        starting_node = [starting_node[0] + direction[0], starting_node[1] + direction[1]]
    return points


csv = pd.read_csv('data/Receiving_Plays/play116-game2021010301.csv')
home, away, balls = extract_one_game(csv)


receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
punt_returner = returner(csv, receive_frame)
returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))

# pixels_array = []
pixels, x_spacing, y_spacing = get_pixels([120, 53], [121, 54])
# print(pixels)
# get_neighbours(pixels[0][0], x_spacing, y_spacing)
# for frame in range(receive_frame+1, receive_frame+10):
frame = receive_frame + 5
pixel_values = assign_pixel_values(frame, home, away, pixels)
    # pixels_array.append(pixel_values)

# print(pixels_array)

fig, ax = plt.subplots()

points = a_star_search(balls[frame], pixels, pixel_values, x_spacing, y_spacing)
xs = []
ys = []
for point in points:
    xs.append(point[0])
    ys.append(point[1])
ax.plot(xs, ys, 'k-')
ax.scatter(points[-1][0], points[-1][1], marker='x', c='r')

ax.scatter(balls[frame][0], balls[frame][1], c='k', s=0.9)
# print(pixels_array[frame])
# pixel_values = assign_pixel_values(frame, home, away, pixels)
for player in home:
    ax.scatter(home[player][frame][0], home[player][frame][1], c='b')
for player in away:
    ax.scatter(away[player][frame][0], away[player][frame][1], c='r')
# ax.scatter(returner_pos[frame][0], returner_pos[frame][1], c='pink', s=0.8)
image = ax.imshow(np.flipud(pixel_values.T), extent=(0, 120, 0, 53), cmap='bwr', alpha=0.4)

plt.show()
