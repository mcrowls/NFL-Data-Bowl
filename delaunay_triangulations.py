from operator import index
from re import M
from turtle import up, update
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skspatial.objects import Line
from skspatial.objects import Point
import math
from players import Player
from window import Window


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


def get_points_of_defenders(defenders, index):
    return [defender.getxyloc(index) for defender in defenders]


def get_lines_from_delaunay(triangles,defenders,frame):
    #Splitting the delaunay triangles into their sides
    index_pairs = []
    #array saving which triangle each window belongs to
    delaunay_triangles = []
    for idx,tri in enumerate(triangles.simplices):
        index_pairs.append(tri[0:2])
        index_pairs.append(tri[1:])
        index_pairs.append(tri[::2])
        #each edge comes from a specific triangles, record those triangle indexes
        delaunay_triangles.append(idx)
        delaunay_triangles.append(idx)
        delaunay_triangles.append(idx)

    #finding the defenders at the end of each line
    defender_pairs = []
    for pair in index_pairs:
        defender_pairs.append([defenders[pair[0]].getxyloc(frame),defenders[pair[1]].getxyloc(frame)])

    #finding the equally spaced points between each defender pair
    points = []
    windows = []
    for pair in defender_pairs:
        #21 is arbitrary, but then the head of the list is removed because it's on top of a defender
        line =  np.linspace(pair[0],pair[1],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0]))
    #assigning each window their triangle
    for t,w in list(zip(delaunay_triangles,windows)):
        windows[windows.index(w)].triangle = [t]
        
    #finding duplicate windows, where they have the same edges, combining those two windows and removing the other window
    #because some windows can belong to two triangles
    for window in windows:
        #print("Current window",window.triangle,list(np.sort(window.points,axis=None)))
        #g = input("")
        for other_window in windows:
            #print("Other window",other_window.triangle,list(np.sort(other_window.points,axis=None)))
            #g = input("")
            if other_window == window:
                continue
            #if np.array_equal(np.sort(window.points,axis=None),np.sort(other_window.points,axis=None)):
            if np.allclose(np.sort(window.points,axis=None),np.sort(other_window.points,axis=None)):
                #print("Found duplicate window",other_window.triangle,list(np.sort(other_window.points,axis=None)))
                window.triangle.append(other_window.triangle[0])
                windows.remove(other_window)
    #print("WINDOWS LENGTH",len(windows))
    points = np.array(points)
    return np.reshape(points,(-1,2)), windows

#calculating the arrival time of each defender to each point in the window, but only keeping the min time
#These comments came from the competition code, detailing how to calculate the time penalty each blocker imposes on a defender
  #   1. Create a straight line between the defender and the target location
  #   2. Find the perpendicular projection of each blocker onto the line from Step 1
  #   3. If this projection does not lie in between the defender and the target, then penalty = 0
  #      Else use a Gaussian kernel with StdDev = the distance between the defender and the blocker's 
  #      perpendicular projection and x = the blocker's distance away from the perpendicular projection 
  #      to obtain the time penalty for the defender based on the blocker's position (multiplied by the)
  #      max_time_penalty parameter
def get_arrival_times(windows,defenders, blockers, frame):
    times = []
    updated_windows = []
    for window in windows:
        optimal_time = float("-inf")
        optimal_point = []
        for p in window.points:
            min_dist = float('inf')
            defender_speed = 0
            for d in defenders:
                dist = np.linalg.norm(d.getxyloc(frame)-p)
                #finding the closest defender to the point, and saving their speed
                if dist < min_dist:
                    min_dist = dist
                    defender_speed = d.speed
            total_penalty = 0
            for b in blockers:
                #path connecting defender to target point
                line = Line.from_points(Point(d.getxyloc(frame)),Point(p))
                #projected point of blocker onto path
                projected_point = line.project_point(b.getxyloc(frame))
                defender_to_projection = np.linalg.norm(d.getxyloc(frame)-projected_point)
                projection_to_target = min_dist - defender_to_projection
                blocker_to_projection = np.linalg.norm(b.getxyloc(frame)-projected_point)
                #if blocker projection is not between defender and target, penalty is 0
                #Gaussian kernel uses a weighting term of 5, not sure why, the code used it before with no explanation
                penalty = 0 if (defender_to_projection >= min_dist or projection_to_target >= min_dist) else 5 * (1 / ((defender_to_projection) * math.sqrt(2*math.pi))) * math.exp(-(1/2) * (blocker_to_projection/(defender_to_projection))**2)
                total_penalty += penalty
        
            time = min_dist / defender_speed + total_penalty
            times.append(time)
            if time > optimal_time:
                optimal_time = time
                optimal_point = p
        updated_windows.append(Window(window.points,time,optimal_point,window.triangle))
    return times, updated_windows

#finding the node (window) with the lowest f value, but may need to change depending on whether g and h need to be minimised or maximised
def find_lowest_f_node(nodes):
    f = float('inf')
    n = None
    for node in nodes:
        if node.f < f:
            f = node.f
            n = node
    return n 


def create_window_neighbors(windows):
    for window in windows:
        #need to find the other windows in the same triangle as the current window
        #windows either have 2 or 4 neighbors
        for other_window in windows:
            if other_window ==  window:
                continue
            for triangle in window.triangle:
                if triangle in other_window.triangle and other_window.optimal_point[0]<= window.optimal_point[0]:
                    window.neighbors.append(other_window)
    return windows


def get_optimal_path(windows,carrier,end):
    #find the closest windows to the carrier
    min_dist = float('inf')
    start_window = None
    #assuming end is an x y point, need to find the window closest to the end
    min_dist_end = float('inf')
    end_window = None
    for window in windows:
        #the start window is the closest window to the returner
        if np.linalg.norm(carrier-window.optimal_point) < min_dist:
            min_dist = np.linalg.norm(carrier-window.optimal_point)
            start_window = window
        #the end window is the closest window to the end point
        if np.linalg.norm(end-window.optimal_point) < min_dist_end:
            min_dist_end = np.linalg.norm(end-window.optimal_point)
            end_window = window
    closed_list = []
    open_list = []


    #append closest window point from carrier to open with an f score of 0
    open_list.append(start_window)
    while len(open_list) > 0:
        current_node = find_lowest_f_node(open_list)
        open_list.remove(current_node)
        closed_list.append(current_node)

        #if the current node is the end node, we're done
        if np.array_equal(current_node.optimal_point,end_window.optimal_point):
            print("Done")
            for i in closed_list:
                print(i.optimal_point)
            #! return the closed list, is this the correct list to return?
            return closed_list
        
        #for all the neighbors for a window, check if they are already in the closed list, if so, do nothing
        for neighbor in current_node.neighbors:
            if neighbor in closed_list:
                continue

            #if the neighbor is not in the closed list, need to calculate the heuristic

            #What heuristic should be used for the windows? Distance or time?
            neighbor.g = current_node.g + np.linalg.norm(neighbor.optimal_point - current_node.optimal_point)
            #neighbor.g = current_node.g + neighbor.optimal_time

            #h is the distance from the neighbor node to the end, only in the x direction
            neighbor.h = np.linalg.norm(neighbor.optimal_point[0] - end[0])
            neighbor.f = neighbor.g + neighbor.h

            #if this neighbor is in the open list already, and it's g value in the open list is less than the g value just calculated, do nothing
            #because the neighbor in the open list is better
            if neighbor in open_list:
                if neighbor.g >= open_list[open_list.index(neighbor)].g:
                    continue
            #else, add it to the open list
            #! I don't think this overrides the neighbor in the open list if the g value is in fact less 
            open_list.append(neighbor)
    return closed_list

# Find defensive locations in each frame
# Need to find the team on the ball and the defensive team
def get_defensive_locations(playpath_):
    csv = pd.read_csv(playpath_)
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
        else:
            defenders.append(Player(player, player_csv['x'], player_csv['y'], team, 0.6))
    return attackers, defenders