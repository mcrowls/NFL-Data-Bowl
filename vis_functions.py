import matplotlib

from delaunay_triangulations import Player, get_points_of_defenders, returner, get_lines_from_delaunay, \
    get_arrival_times

matplotlib.use("TkAgg")
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from scipy.spatial import Delaunay


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




def animate_return(csv):
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
        elif team != "football":
            defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
    ball_df = csv.sort_values(['frameId'], ascending=True)
    ball_df = ball_df[ball_df.team == "football"][receive_frame:]
    balls = list(zip(ball_df.x.tolist(), ball_df.y.tolist()))
    fig, ax = drawPitch(100, 53.3)

    #DO CALC BEFORE TO SPEED UP VISUALS
    points_def = []
    points_off = []
    lines = []
    times = []

    for frame in range(size):
        points_def.append(np.array(get_points_of_defenders(defenders, frame)))
        points_off.append(np.array(get_points_of_defenders(attackers, frame)))
        tri = Delaunay(points_def[frame])
        lines.append(get_lines_from_delaunay(tri, points_def[frame]))
        times.append(get_arrival_times(lines[frame], points_def[frame], points_off[frame]))




    for frame in range(size):
        # points_def = np.array(get_points_of_defenders(defenders, frame))
        # points_off = np.array(get_points_of_defenders(attackers, frame))
        # tri = Delaunay(points_def)
        # lines = get_lines_from_delaunay(tri, points_def)
        # times = get_arrival_times(lines, points_def, points_off)
        # PLOT EVERYTHING
        defensive, = ax.plot(points_def[frame][:, 0], points_def[frame][:, 1], 'o', markersize=10, markerfacecolor="r",
                             markeredgewidth=1, markeredgecolor="white",
                             zorder=7, label='Defenders')
        offensive, = ax.plot(points_off[frame][:, 0], points_off[frame][:, 1], 'o', markersize=10, markerfacecolor="b",
                             markeredgewidth=1, markeredgecolor="white",
                             zorder=7, label='Attackers')
        ball, = ax.plot(balls[frame][0], balls[frame][1], 'o', markersize=8, markerfacecolor="black", markeredgewidth=1, markeredgecolor="white",
                    zorder=7)

        #triang = ax.triplot(*points_def.T, tri.simplices, color="black")

        p = ax.scatter(lines[frame][:, 0], lines[frame][:, 1], c=times[frame], cmap="YlOrRd", marker="s", s=5)


        if frame < size - 1:
            plt.pause(0.05)
            p.remove()
            offensive.remove()
            defensive.remove()
            ball.remove()
            #triang[0].remove()
            #triang[1].remove()

    plt.show()

#
# def animate_one_play(df, delaunay=False):
#     fig, ax = drawPitch(100, 53.3)
#
#     home, away, balls = extract_one_game(df)
#
#     team_left, = ax.plot([], [], 'o', markersize=10, markerfacecolor="r", markeredgewidth=1, markeredgecolor="white",
#                          zorder=7)
#     team_right, = ax.plot([], [], 'o', markersize=10, markerfacecolor="b", markeredgewidth=1, markeredgecolor="white",
#                           zorder=7)
#     ball, = ax.plot([], [], 'o', markersize=8, markerfacecolor="black", markeredgewidth=1, markeredgecolor="white",
#                     zorder=7)
#     #lines, = ax.triplot([],[])
#
#
#     #drawings = [team_left, team_right, ball, lines]
#     drawings = [team_left, team_right, ball]
#
#     def init():
#         team_left.set_data([], [])
#         team_right.set_data([], [])
#         ball.set_data([], [])
#
#         #lines.set_data([],[])
#
#         return drawings
#
#     def draw_teams(i):
#         X = []
#         Y = []
#         for k, v in home.items():
#             x, y = v[i]
#             X.append(x)
#             Y.append(y)
#         team_left.set_data(X, Y)
#
#         X = []
#         Y = []
#         for k, v in away.items():
#             x, y = v[i]
#             X.append(x)
#             Y.append(y)
#         team_right.set_data(X, Y)
#
#     def animate(i):
#         draw_teams(i)
#         x, y = balls[i]
#         ball.set_data([x, y])
#
#         # if delaunay:
#         #     csv = df
#         #     receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
#         #     punt_returner = returner(csv, receive_frame)
#         #     attacking_team = csv[csv['displayName'] == punt_returner]['team'].iloc[0]
#         #     attackers = []
#         #     defenders = []
#         #     for player in np.unique(csv['displayName']):
#         #         player_csv = csv[csv['displayName'] == player]
#         #         size = np.shape(player_csv)[0]
#         #         team = csv[csv['displayName'] == player]['team'].iloc[0]
#         #         if team == attacking_team:
#         #             attackers.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
#         #         else:
#         #             defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
#         #
#         #     points_def = np.array(get_points_of_defenders(defenders, i))
#         #     tri = Delaunay(points_def)
#         #     #lines.set_data(*points_def.T, tri.simplices)
#         #     ax.triplot(points_def[:,0], points_def[:,1], tri.simplices)
#         return drawings
#     # !May take a while!
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=len(balls), interval=100, blit=False)
#
#
#     #plt.show()
#     return anim #HTML(anim.to_html5_video())
