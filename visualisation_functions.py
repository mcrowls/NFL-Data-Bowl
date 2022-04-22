import matplotlib
from delaunay_triangulations import Player, create_window_neighbors, frechet_distance, get_optimal_path, get_points_of_defenders, returner, get_lines_from_delaunay, \
    get_arrival_times, get_defensive_locations, boundary_windows, get_lines_from_sidelines
from players import get_player_speed
try:
    matplotlib.use("TkAgg")
except:
    matplotlib.use('Qt5Agg')#WebAgg
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import statistics
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
from helpers import get_play_description_from_number, inputpath, playpath, visoutputpath, play_folderpath, create_new_folder
import time
from matplotlib import animation
from IPython.display import HTML
import os
import ipywidgets as widgets
from IPython.display import display,clear_output
from ipywidgets import Output, Button
import functools

def drawPitch(width, height, color="black"):
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, width + 30), ylim=(-15, height + 5))
    plt.axis('off')

    # Grass around pitch
    rect = patches.Rectangle((-10, -5), width + 40, height + 10, linewidth=1, facecolor='#ddede5', capstyle='round')
    ax.add_patch(rect)
    ###################

    # Pitch boundaries
    rect = plt.Rectangle((0, 0), width + 20, height, ec=color, fc="None", lw=2)
    ax.add_patch(rect)
    ###################
    # vertical lines - every 5 yards
    for i in range(21):
        plt.plot([10 + 5 * i, 10 + 5 * i], [0, height], c="black", lw=2)
    ###################

    # distance markers - every 10 yards
    for yards in range(10, width, 10):
        yards_text = yards if yards <= width / 2 else width - yards
        # top markers
        plt.text(10 + yards - 2, height - 7.5, yards_text, size=20, c="black")
        # botoom markers
        plt.text(10 + yards - 2, 7.5, yards_text, size=20, c="black", rotation=180)
    ###################
        # yards markers - every yard
        # bottom markers
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [1, 3], color="black", lw=3)

        # top markers
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [height - 1, height - 3], color="black", lw=3)

        # middle bottom markers
        y = (height - 18.5) / 2
        for x in range(20):
            for j in range(1, 5):
                plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [y, y + 2], color="black", lw=3)
    # middle top markers
    for x in range(20):
        for j in range(1, 5):
            plt.plot([10 + x * 5 + j, 10 + x * 5 + j], [height - y, height - y - 2], color="black", lw=3)
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
    plt.plot([10 + 2, 10 + 2], [y, y + 3], c="black", lw=2)

    # right
    plt.plot([width + 10 - 2, width + 10 - 2], [y, y + 3], c="black", lw=2)
    ###################

    # draw goalpost
    goal_width = 6  # yards
    y = (height - goal_width) / 2
    # left
    plt.plot([0, 0], [y, y + goal_width], "-", c="y", lw=10, ms=20)
    # right
    plt.plot([width + 20, width + 20], [y, y + goal_width], "-", c="black", lw=10, ms=20)

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

def process_frames(csv, delaunay=False, print_status=False):
    play_direction = csv["playDirection"].iloc[0]
    receive_frame = csv[csv['event'] == 'punt_received']['frameId'].iloc[0]
    punt_returner = returner(csv, receive_frame)
    attacking_team = csv[csv['displayName'] == punt_returner]['team'].iloc[0]
    attackers = []
    defenders = []

    for player in np.unique(csv['displayName']):
        player_csv = csv[csv['displayName'] == player][receive_frame:]
        #size = np.shape(player_csv)[0]
        size = 20
        team = csv[csv['displayName'] == player]['team'].iloc[0]
        if team == attacking_team:
            attackers.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
        elif team != "football":
            defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))

    returner_pos = csv[csv['displayName'] == punt_returner][receive_frame:]
    returner_pos = list(zip(returner_pos.x.tolist(), returner_pos.y.tolist()))
    returner_speed = get_player_speed(punt_returner)

    ball_df = csv.sort_values(['frameId'], ascending=True)
    ball_df = ball_df[ball_df.team == "football"][receive_frame:]
    balls = list(zip(ball_df.x.tolist(), ball_df.y.tolist()))

    #DO CALC BEFORE TO SPEED UP VISUALS
    points_def = []
    points_off = []
    lines = []
    times = []
    optimal_points = []
    optimal_paths = []
    top_windows = []
    right_windows = []
    left_windows = []
    optimal_paths = []
    all_windows = []
    frechets = []
    start_time = time.time()
    # Get data before for smooth animation

    yardage_gained = returner_pos[0][0] - returner_pos[-1][0]

    for frame in range(size):
        #if the returner goes out of bounds then stop calculating
        if returner_pos[frame][1] <= 0 or returner_pos[frame][1] >= 53:
            print("Returner went out of bounds, stopping calculations")
            yardage_gained = returner_pos[0][0] - returner_pos[frame][0]
            size = frame + 1
            break
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
        arrival_time, windows = get_arrival_times(windows,side_windows,defenders,attackers,frame,returner_pos[frame])
        times.append(arrival_time)

        #Calculate the optimal path through the windows
        windows = create_window_neighbors(windows)
        optimal_path = get_optimal_path(windows,[returner_pos[frame][0],returner_pos[frame][1]],[10,25], returner_speed)
        optimal_path_points = []
        for window in optimal_path:
            optimal_path_points.append(window.optimal_point)

        if abs(returner_pos[frame][0] - returner_pos[-1][0]) > 1:
            frechetDistance = frechet_distance(np.array(returner_pos[frame:]).reshape(-1,2),optimal_path_points)

            #frechetDistance = frechetDistance * 5 /
            frechets.append(frechetDistance)

        optimal_paths.append(np.reshape(optimal_path_points,(-1,2)))
        o = []
        l = []
        for w in windows:
            o.append(w.optimal_point)
            l.append(w.points)

        l = np.array(l)
        lines.append(np.reshape(l,(-1,2)))
        optimal_points.append(o)
        all_windows.append(windows)
        print("Processed frame", frame+1, "/",size,"||",round(((frame+1)/size)*100),"%")

    end_time = time.time()
    if print_status:
        print("Took",round(end_time-start_time,2),"s to process",size,"frames")
    return size, returner_pos, points_def, points_off, balls, lines, times, optimal_paths, optimal_path_points, windows, all_windows, optimal_points, play_direction,frechets,yardage_gained

def animate_return(csv, delaunay=False, print_status=False, use_funcanim=False, outpath=visoutputpath, playname=play_folderpath):
    fig, ax = drawPitch(100, 53.3)
    anim_values = []
    size,returner_pos,home,away,balls,lines,times,optimal_paths,optimal_path_points,windows,all_windows,optimal_points,play_direction,frechets,yardage_gained = process_frames(csv, delaunay, print_status)
    anim_values.extend([returner_pos, home,away,balls,lines,times])
    ax.plot(np.array(returner_pos).reshape(-1,2)[:,0],np.array(returner_pos).reshape(-1,2)[:,1])

    for frame in range(size):
        # PLOT EVERYTHING
        #optimal = ax.scatter(optimal_paths[frame][:,0],optimal_paths[frame][:,1],marker="*",c="pink",zorder=17)
        
        """neighborlines = []
        for window in all_windows[frame]:
            if window.triangle == []:
                for n in window.neighbors:
                    neighborlines.append(ax.plot([window.optimal_point[0],n.optimal_point[0]],[window.optimal_point[1],n.optimal_point[1]],color="red",))
            else:
                for n in window.neighbors:
                    neighborlines.append(ax.plot([window.optimal_point[0],n.optimal_point[0]],[window.optimal_point[1],n.optimal_point[1]],color="red"))"""
        
        arrow, = ax.plot(optimal_paths[frame][:,0],optimal_paths[frame][:,1],c="green",zorder=15,linewidth=4.5)
        #retur = ax.text(returner_pos[frame][0]-0.5, returner_pos[frame][1]-0.5, 'R', zorder=15, c="pink")

        returner_line, = ax.plot([returner_pos[frame][0], returner_pos[frame][0]], [0, 53.3], "--", zorder=6, c="black")
        returner_path, = ax.plot([returner_pos[frame][0], returner_pos[frame][0]], [0, 53.3], "-", zorder=5, c="gray", linewidth=5)
        returner_pos_, = ax.plot([returner_pos[frame][0], returner_pos[frame][0]], [0, 53.3], 'o', markersize=13, markerfacecolor="gray", markeredgewidth=1, markeredgecolor="white", zorder=9)
        defensive, = ax.plot(home[frame][:, 0], home[frame][:, 1], 'o', markersize=10, markerfacecolor="r",
                            markeredgewidth=1, markeredgecolor="white",
                            zorder=5, label='Defenders')
        offensive, = ax.plot(away[frame][:, 0], away[frame][:, 1], 'o', markersize=10, markerfacecolor="b",
                            markeredgewidth=1, markeredgecolor="white",
                            zorder=5, label='Attackers')
        ball, = ax.plot(balls[frame][0], balls[frame][1], 'o', markersize=8, markerfacecolor="black", markeredgewidth=1, markeredgecolor="white",
                zorder=10)

        w = ax.scatter(np.array(optimal_points[frame])[:,0],np.array(optimal_points[frame])[:,1],c = "black",marker="x",zorder=16)
        if delaunay:
            p = ax.scatter(lines[frame][:, 0], lines[frame][:, 1], c=times[frame], cmap="YlOrRd", marker="s", s=5, zorder=15)
        
        plt.savefig(f"visualisations/{playname}_frame{frame}.png", format="png")
        if frame < size - 1:
            plt.pause(0.20)
            if delaunay:
                p.remove()
            returner_line.remove()
            returner_path.remove()
            returner_pos_.remove()
            offensive.remove()
            defensive.remove()
            ball.remove()
            #retur.remove()
            w.remove()
            arrow.remove()
            #optimal.remove()
            #for n in neighborlines:
                #n[0].remove()

    plt.savefig(f"visualisations/{playname[:len(playname) - 4]}.png", format="png")
    plt.show()

def visualise_play(playpath_, changeFigsize=False, outpath=visoutputpath, playname=play_folderpath):
    if changeFigsize:
        plt.rcParams['figure.figsize'] = [18, 10]
    csv = pd.read_csv(playpath_)
    animate_return(csv, delaunay=True, outpath=outpath, playname=playname)

"""
Get yardage and frechet distance for one play
"""
def process_play(playpath_,playname,output_results):
    csv = pd.read_csv(playpath_)
    try:
        size,returner_pos,home,away,balls,lines,times,optimal_paths,optimal_path_points,windows,all_windows,optimal_points,play_direction,frechets,yardage_gained = process_frames(csv, True, True)
        median_deviation = np.median(np.array(frechets))
        mean_deviation = np.mean(np.array(frechets))
        d = {'play':[playname],'yardage':[yardage_gained],'median_deviation':median_deviation,'mean_deviation':mean_deviation}
        df = pd.DataFrame(data=d)
        df.to_csv(output_results, mode='a', header=False)
    #If the play crashes just output nothing
    except Exception as e:
        d = {'play':[playname],'yardage':None,'median_deviation':None,'mean_deviation':None}
        df = pd.DataFrame(data=d)
        df.to_csv(output_results, mode='a', header=False)
    
    

# Draw the delaunay triangles frame by frame
def visualise_delaunay_play(playpath_, outpath=visoutputpath, playname=play_folderpath, size=40):
    create_new_folder(outpath)
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
        plt.savefig(f"visualisations/{playname[:len(playname) - 4]}_frame{frame}.png", format="png")
        if frame < size-1:
            plt.pause(0.05)
            ax.clear()
            fig.clear()
    plt.show()

def animate_one_play(home, away, balls, return_line, lines,times, play_direction, outpath=visoutputpath, playname=play_folderpath):
    fig, ax = drawPitch(100, 53.3)
    team_left, = ax.plot([], [], 'o', markersize=12, markerfacecolor="r", markeredgewidth=1, markeredgecolor="white", zorder=7)
    team_right, = ax.plot([], [], 'o', markersize=12, markerfacecolor="b", markeredgewidth=1, markeredgecolor="white", zorder=7)
    ball, = ax.plot([], [], 'o', markersize=8, markerfacecolor="black", markeredgewidth=1, markeredgecolor="white", zorder=7)
    returner_line, = ax.plot([],[], "--", zorder=5, c="black")
    returner_path, = ax.plot([], [], "-", zorder=5, c="gray", linewidth=5)
    returner_pos, = ax.plot([], [], 'o', markersize=13, markerfacecolor="gray", markeredgewidth=1, markeredgecolor="white", zorder=9)
    p = ax.scatter(lines[0][:, 0], lines[0][:, 1], c=times[0], cmap="YlOrRd", marker="s", s=5, zorder=5)
    drawings = [team_left, team_right, ball, returner_line, returner_path, returner_pos, p]

    def init():
        team_left.set_data([], [])
        team_right.set_data([], [])
        ball.set_data([], [])
        returner_line.set_data([],[])
        if play_direction == 'left':
            returner_path.set_data([120 - x[0] for x in return_line], [160/3 -x[1] for x in return_line])
        else:
            returner_path.set_data([x[0] for x in return_line], [x[1] for x in return_line])
        returner_pos.set_data([],[])

        return drawings

    def draw_teams(i):
        if play_direction == 'left':
            X = []
            Y = []
            for x,y in home[i]:
                X.append(120-x)
                Y.append(160/3 - y)
            team_left.set_data(X, Y)

            X = []
            Y = []
            for x,y in away[i]:
                X.append(120 - x)
                Y.append(160/3 - y)
            team_right.set_data(X, Y)

            returner_pos.set_data(120 - return_line[i][0], 160/3 - return_line[i][1])
        else:
            X = []
            Y = []
            for x,y in home[i]:
                X.append(x)
                Y.append(y)
            team_left.set_data(X, Y)

            X = []
            Y = []
            for x,y in away[i]:
                X.append(x)
                Y.append(y)
            team_right.set_data(X, Y)

            returner_pos.set_data(return_line[i][0], return_line[i][1])

    def animate(i):
        if play_direction == 'left':
            draw_teams(i)

            x, y = balls[i]
            ball.set_data([120 - x, 160/3 - y])
            x, y = return_line[i]
            returner_line.set_data([[120 - x, 120 - x], [0,53.3]])
            X = []
            Y = []
            for x,y in lines[i]:
                X.append(120 - x)
                Y.append(160/3 - y)

            p.set_offsets(np.transpose([X,Y]))

            return drawings
        else:
            draw_teams(i)

            x, y = balls[i]
            ball.set_data([x, y])
            x, y = return_line[i]
            returner_line.set_data([[x, x], [0,53.3]])
            X = []
            Y = []
            for x,y in lines[i]:
                X.append(x)
                Y.append(y)

            p.set_offsets(np.transpose([X,Y]))

            return drawings
    # !May take a while!
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(balls), interval=100, blit=False)
    
    writergif = animation.PillowWriter(fps=30) 
    anim.save(outpath+playname[:-4]+".gif", writer=writergif)
    return HTML(anim.to_html5_video())

def visualise_play_FuncAnimation(play=playpath, outpath=visoutputpath, playname=play_folderpath):
    create_new_folder(outpath)
    plt.rcParams['figure.figsize'] = [18, 10]
    anim_values = []
    if play == playpath:
        csv = pd.read_csv(playpath)
    else:
        csv = pd.read_csv(inputpath+"receiving_plays/"+play)
    size,returner_pos,home,away,balls,lines,times,ox,oy,optimal_path,optimal_path_points,windows,optimal_points,play_direction = process_frames(csv, delaunay=True)
    anim_values.extend([returner_pos, home,away,balls,lines,times,ox,oy, play_direction])
    animate_one_play(anim_values[1],anim_values[2],anim_values[3],anim_values[0],anim_values[4],anim_values[5], anim_values[8], outpath, playname)

#OLD visualise_delaunay_play(playpath)
#visualise_play_FuncAnimation(playpath)
#visualise_play(playpath, changeFigsize=True)