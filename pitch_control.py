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


# returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
# returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))

def get_pixels(pitch_size, grid_size):
    xs = np.linspace(0, pitch_size[0], grid_size[0])
    ys = np.linspace(0, pitch_size[1], grid_size[1])
    pixels = []
    for x in xs:
        row = []
        for y in ys:
            row.append([x, y])
        pixels.append(row)
    return pixels


def get_closest(point, points):
    # Want to add in speed
    distances = []
    for other_point in points:
        distances.append(np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2))
    return distances.index(np.min(distances))


def assign_pixel_values(frame, home, away, pixels):
    # print(np.shape(pixels))
    values = np.zeros((120, 53))
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


csv = pd.read_csv('data/Receiving_Plays/play116-game2021010301.csv')
home, away, balls = extract_one_game(csv)


receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
punt_returner = returner(csv, receive_frame)
returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))

pixels_array = []
pixels = get_pixels([120, 53], [120, 53])
for frame in range(np.max(csv['frameId'])):
    pixel_values = assign_pixel_values(frame, home, away, pixels)
    pixels_array.append(pixel_values)

# print(pixels_array)


fig, ax = plt.subplots()
for frame in range(np.max(csv['frameId'])):
    print(pixels_array[frame])
    # pixel_values = assign_pixel_values(frame, home, away, pixels)
    for player in home:
        ax.scatter(home[player][frame][0], home[player][frame][1], c='b')
    for player in away:
        ax.scatter(away[player][frame][0], away[player][frame][1], c='r')
    ax.imshow(np.flipud(pixels_array[frame].T), extent=(0, 120, 0, 53), cmap='bwr', alpha=0.5)
    plt.pause(0.0001)
    ax.clear()
plt.show()
