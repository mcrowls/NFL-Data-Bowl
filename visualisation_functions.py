import matplotlib
from delaunay_triangulations import Player, create_window_neighbors, get_lines_from_sidelines, get_optimal_path, get_points_of_defenders, returner, get_lines_from_delaunay, \
    get_arrival_times, get_defensive_locations
matplotlib.use("TkAgg")
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
from helpers import get_play_description_from_number, inputpath, playpath
import time
from side_windows import *



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




def animate_return(csv, delaunay=False):

    receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
    punt_returner = returner(csv, receive_frame)
    attacking_team = csv[csv['displayName'] == punt_returner]['team'].iloc[0]
    attackers = []
    defenders = []

    for player in np.unique(csv['displayName']):
        player_csv = csv[csv['displayName'] == player][receive_frame:]
        #size = np.shape(player_csv)[0]
        size = 9
        team = csv[csv['displayName'] == player]['team'].iloc[0]
        if team == attacking_team:
            attackers.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
        elif team != "football":
            defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))

    returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
    returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))
    

    ball_df = csv.sort_values(['frameId'], ascending=True)
    ball_df = ball_df[ball_df.team == "football"][receive_frame:]
    balls = list(zip(ball_df.x.tolist(), ball_df.y.tolist()))
    fig, ax = drawPitch(100, 53.3)

    #DO CALC BEFORE TO SPEED UP VISUALS
    points_def = []
    points_off = []
    lines = []
    times = []
    optimal_points = []
    optimal_paths = []
    bound_points_x =[]
    bound_points_y = []
    outer_layer_x = []
    outer_layer_y = []
    top_windows = []
    right_windows = []
    left_windows = []
    optimal_paths = []
    start_time = time.time()
    # Get data before for smooth animation
    for frame in range(size): 
        points_def.append(np.array(get_points_of_defenders(defenders, frame)))
        points_off.append(np.array(get_points_of_defenders(attackers, frame)))
        if delaunay:
            bounds = ConvexHull(points_def[frame]).vertices
            top, right, left = boundary_windows(points_def[frame][bounds], returner_pos[frame][0])
            top_windows.append(top)
            right_windows.append(right)
            left_windows.append(left)

        #Get delaunay triangles and arrival times in the windows
        tri = Delaunay(points_def[frame])
        _, windows = get_lines_from_delaunay(tri,defenders,frame)
        _,side_windows = get_lines_from_sidelines(top,left,right,returner_pos[frame])
        arrival_time, windows = get_arrival_times(windows,side_windows,defenders,attackers,frame)
        times.append(arrival_time)

        #Calculate the optimal path through the windows
        windows = create_window_neighbors(windows)
        optimal_path = get_optimal_path(windows,[returner_pos[frame][0],returner_pos[frame][1]],[10,25])
        optimal_path_points = []
        for window in optimal_path:
            optimal_path_points.append(window.optimal_point)
        optimal_paths.append(np.reshape(optimal_path_points,(-1,2)))
        o = []
        l = []
        for w in windows:
            o.append(w.optimal_point)
            l.append(w.points)

        #for s in side_windows:
         #   l.append(s.points)

        l = np.array(l)
        lines.append(np.reshape(l,(-1,2)))
        optimal_points.append(o)
        print("Processed frame", frame+1, "/",size,"||",round(((frame+1)/size)*100),"%")
    
    end_time = time.time()
    print("Took",round(end_time-start_time,2),"s to process",size,"frames")
    for frame in range(size):
        # PLOT EVERYTHING

        #The points on the optimal path
        optimal = ax.scatter(optimal_paths[frame][:,0],optimal_paths[frame][:,1],marker="*",c="pink",zorder=17)

        #The lines on the optimal path
        arrow, = ax.plot(optimal_paths[frame][:,0],optimal_paths[frame][:,1],c="black")
        
        for window in windows:
            """THESE ARE DEBUGGING VISUALISTIONS"""
            #These show the coodinates of optimal points
            #t = str(round(window.optimal_point[0],1))+" "+str(round(window.optimal_point[1],1))
            #ax.text(window.optimal_point[0]-0.5,window.optimal_point[1]-0.5,t)
            #t = str(round(window.start[0],1))+" "+str(round(window.start[1],1))
            #ax.text(window.optimal_point[0]+0.5,window.optimal_point[1],t,c="orange")
            #if len(window.end) >0 :
                #ax.text(window.end[0]+0.5,window.end[1]+0.5,"E")
            
            #These draw lines between windows and their neighbors for degugging purposes
            if window.triangle == []:
                for n in window.neighbors:
                    ax.arrow(window.optimal_point[0],window.optimal_point[1],n.optimal_point[0]-window.optimal_point[0],n.optimal_point[1]-window.optimal_point[1],color="blue",)
            else:
                for n in window.neighbors:
                    ax.arrow(window.optimal_point[0],window.optimal_point[1],n.optimal_point[0]-window.optimal_point[0],n.optimal_point[1]-window.optimal_point[1],color="orange")

        #The returner
        retur = ax.text(returner_pos[frame][0]-0.5, returner_pos[frame][1]-0.5, 'R', zorder=15, c="pink")

        #The y line of the returner
        return_line, = ax.plot([returner_pos[frame][0], returner_pos[frame][0]], [0, 53.3], "--", zorder=5, c="black")

        defensive, = ax.plot(points_def[frame][:, 0], points_def[frame][:, 1], 'o', markersize=10, markerfacecolor="r",
                             markeredgewidth=1, markeredgecolor="white",
                             zorder=5, label='Defenders')
        offensive, = ax.plot(points_off[frame][:, 0], points_off[frame][:, 1], 'o', markersize=10, markerfacecolor="b",
                             markeredgewidth=1, markeredgecolor="white",
                             zorder=5, label='Attackers')
        ball, = ax.plot(balls[frame][0], balls[frame][1], 'o', markersize=8, markerfacecolor="black", markeredgewidth=1, markeredgecolor="white",
                    zorder=10)
        
        #The optimal points on each window
        w = ax.scatter(np.array(optimal_points[frame])[:,0],np.array(optimal_points[frame])[:,1],c = "black",marker="x",zorder=16)

        # triang = ax.triplot(*points_def.T, tri.simplices, color="black")
        if delaunay:
            top_points = []
            left_points = []
            right_points = []
            #The heatmap of the arrival times on all the windows
            p = ax.scatter(lines[frame][:, 0], lines[frame][:, 1], c=times[frame], cmap="YlOrRd", marker="s", s=5, zorder=15)
            #The sideline points
            """for point in top_windows[frame]:
                top_points+=ax.plot([point[0], returner_pos[frame][0]], [point[1], point[1]], 'k',zorder=16)
            for point in left_windows[frame]:
                left_points+=ax.plot([point[0], point[0]], [point[1], 53.3], 'k',zorder=16)
            for point in right_windows[frame]:
                right_points+=ax.plot([point[0], point[0]], [point[1], 0 ], 'k',zorder=16)"""

        if frame < size - 1:
            plt.pause(0.20)
            if delaunay:
                p.remove()
                for points in top_points:
                    points.remove()
                for points in left_points:
                    points.remove()
                for points in right_points:
                    points.remove()

            return_line.remove()
            offensive.remove()
            defensive.remove()
            ball.remove()
            retur.remove()
            w.remove()
            optimal.remove()
            arrow.remove()

            #triang[0].remove()
            #triang[1].remove()

    plt.show()

def visualise_play(playpath_, changeFigsize=False):
    if changeFigsize:
        plt.rcParams['figure.figsize'] = [18, 10]
    csv = pd.read_csv(playpath_)
    animate_return(csv, delaunay=True)

# Identify the delaunay triangles within the defensive structure

# Draw the delaunay triangles frame by frame

def visualise_delaunay_play(playpath_, size):
    fig, ax = plt.subplots()
    for frame in range(size):
        attackers, defenders = get_defensive_locations(playpath_)
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



#visualise_delaunay_play(playpath)
visualise_play(playpath)