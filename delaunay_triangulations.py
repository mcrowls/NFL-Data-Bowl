from operator import index
from re import M
from turtle import up, update
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pyrsistent import T
from scipy.spatial import Delaunay
from skspatial.objects import Line
from skspatial.objects import Point
import math
from players import Player
from window import Window


def angle(point1, point2):
    """print(abs(np.arctan((point2[1] - point1[1])/(point2[0] - point1[0]))))
    print(point1,point2)
    print(point2[1] - point1[1])
    print((point2[0] - point1[0]))
    print(" ")"""
    return abs(np.arctan((point2[1] - point1[1])/(point2[0] - point1[0])))


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
        windows.append(Window(line,0,[0,0],start=pair[1],end=pair[0]))
    
    #assigning each window their triangle
    for t,w in list(zip(delaunay_triangles,windows)):
        windows[windows.index(w)].triangle = [t]

    #finding duplicate windows, where they have the same edges, combining those two windows and removing the other window
    #because some windows can belong to two triangles
    for window in windows:
        for other_window in windows:
            if other_window == window:
                continue
            if np.allclose(np.sort(window.points,axis=None),np.sort(other_window.points,axis=None)):
                window.triangle.append(other_window.triangle[0])
                windows.remove(other_window)
    points = np.array(points)
    return np.reshape(points,(-1,2)), windows

#turning the sideline points into window objects
def get_lines_from_sidelines(top,left,right,returner_pos):
    points = []
    windows = []
    for t in top:
        line = np.linspace(t,[returner_pos[0],t[1]],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=t,direction="t"))

    for l in left:
        line = np.linspace(l,[l[0], 53.3],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=l,direction="l"))

    for r in right:
        line = np.linspace(r,[r[0], 0 ],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=r,direction="r"))

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
def get_arrival_times(windows,side_windows,defenders, blockers, frame):
    times = []
    updated_windows = []
    for window in windows+side_windows:
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
        updated_windows.append(Window(window.points,time,optimal_point,window.triangle,window.start,window.end,window.direction))
    return times, updated_windows

#finding the node (window) with the lowest f value, but may need to change depending on whether g and h need to be minimised or maximised
def find_lowest_f_node(nodes):
    f = -float('inf')
    n = None
    for node in nodes:
        if node.f > f:
            f = node.f
            n = node
    return n


def create_window_neighbors(windows):
    for window in windows:
        for other_window in windows:
            if other_window ==  window:
                continue

            #this is to give delaunay windows delaunay neighbors
            #need to find the other windows in the same triangle as the current window
            #windows either have 2 or 4 neighbors
            if len(window.triangle) > 0:
                for triangle in window.triangle:
                    if triangle in other_window.triangle and other_window.optimal_point[0]<= window.optimal_point[0]:
                        window.neighbors.append(other_window)
                
                #giving delaunay windows sideline neighbors
                #if the delaunay window has the same start point as the sideline window, they should be neighbors
                #! But this causes problems because it causes some window neighbors to have other windows in between them
                #! Which is why the check_window_neighbors was created
                if len(other_window.triangle) == 0 and len(window.triangle) ==  1:
                    if (np.array_equal(window.start, other_window.start) ) and other_window.optimal_point[0]<= window.optimal_point[0]:
                        window.neighbors.append(other_window)

            #this is to give sideline windows any neighbors
            else:
                #matching sidelines to delaunay windows
                #if the delaunay window has the same start point as the sideline window, they should be neighbors
                #! But this causes problems because it causes some window neighbors to have other windows in between them
                #! Which is why the check_window_neighbors was created
                if (np.allclose(other_window.start,window.start) or np.array_equal(other_window.end,window.start)) and len(other_window.triangle) == 1 and other_window.optimal_point[0]<= window.optimal_point[0]:
                    window.neighbors.append(other_window)
                
    #Now check for bad neighbors
    windows = check_window_neighbors(windows)
    return windows

#order all left windows, the right one should have the left one as a neighbor
def check_window_neighbors(windows):
    #this gives sideline windows sideline neighbors
    tops = []
    lefts = []
    rights = []
    #create lists of all the sideline window types
    for window in windows:
        if window.direction == "t":
            tops.append(window)
        elif window.direction == "l":
            lefts.append(window)
        elif window.direction == "r":
            rights.append(window)
    #sort them based on the optimal point
    tops.sort(key=lambda x: x.optimal_point[1], reverse=True)
    lefts.sort(key=lambda x: x.optimal_point[0], reverse=True)
    rights.sort(key=lambda x: x.optimal_point[0], reverse=True)

    #give each sideline window it's adjacent sideline windows as neighbors
    for top in tops[1:len(tops)-1]:
        prev = tops[tops.index(top)-1]
        next = tops[tops.index(top)+1]
        top.neighbors.append(prev)
        top.neighbors.append(next)

    tops[0].neighbors.append(tops[1])
    tops[-1].neighbors.append(tops[-2])

    for left in lefts[1:len(lefts)-1]:
        prev = lefts[lefts.index(left)-1]
        next = lefts[lefts.index(left)+1]
        left.neighbors.append(prev)
        left.neighbors.append(next)

    lefts[0].neighbors.append(lefts[1])
    lefts[-1].neighbors.append(lefts[-2])

    for right in rights[1:len(rights)-1]:
        prev = rights[rights.index(right)-1]
        next = rights[rights.index(right)+1]
        right.neighbors.append(prev)
        right.neighbors.append(next)

    rights[0].neighbors.append(rights[1])
    rights[-1].neighbors.append(rights[-2])

    #the first top sideline window should be connected to the nearest left sideline window
    tops[0].neighbors.append(lefts[0])
    lefts[0].neighbors.append(tops[0])

    #the last top sideline window should be connected to the nearest right sideline window
    tops[-1].neighbors.append(rights[0])
    rights[0].neighbors.append(tops[-1])


    #This is an attempt to check that if a window has a neighbor, there is no other window between them
    #It goes through each window's neighbors, checks if there is a window intersecting the line between the window and it's neighbor
    #If there is an intersection, then remove the neighbor
    #However, this doesn't work as planned
    """lines = []
    #Create a line object for all windows
    for window in windows:
        #Sideline windows don't have an end point, so their line's end point is the window's optimal point
        if window.end != []:
            lines.append((Line.from_points(window.start,window.end),window))
        else:
            lines.append((Line.from_points(window.start,window.optimal_point),window))
    
    #for each line's neighbors, check if the line between the line's optimal point and the neighbor's optimal point intersects any window line
    #will know there's an intersect if the intersect point is more to the right than the neighbor's optimal point, so remove that neighbor
    for line in lines:
        #only check outer delaunay windows
        print("CURRENT LINE",line[0],line[1].optimal_point)
        for neighbor in line[1].neighbors:
            #delaunay windows are correctly connected to delaunay windows, don't need to check
            if len(neighbor.triangle) > 0 and len(line[1].triangle) > 0:
                continue

            #! some sideline windows have themselves as neighbors?
            if np.allclose(neighbor.optimal_point,line[1].optimal_point):
                line[1].neighbors.remove(neighbor)
                continue
            
            #now it should only be checking delaunay to sideline window connections
            print("current neighbor", neighbor.optimal_point)
            #create a line between the current window being checked and the current neighbor being checked
            neighbor_line = Line.from_points(line[1].optimal_point,neighbor.optimal_point)
            print("line created between line and neighbor",neighbor_line)
            #now want to check for possible intersections between the neighbor line and any other window
            for line_2 in lines:
                #don't check intersection for the current line being checked
                if line_2 == line:
                    continue
                #don't check intersection for current neighbor being checked
                if line_2[1] == neighbor:
                    continue
                #If the line to check for intersection is behind the current line, don't check it
                #the start and end x values for line_2 (the line to check for potential intersection) should be both greater than the optimal point x to not check line_2
                if line_2[1].start[0] >= line[1].optimal_point[0] and (line_2[1].end == [] or (line_2[1].end != [] and line_2[1].end[0] >= line[1].optimal_point[0])):
                    continue
                print("possible crossing line",line_2[0],line_2[1].optimal_point)
                intersect_point = line_2[0].intersect_line(neighbor_line)
                print("They intersect at",intersect_point)
                #The intersect point should be in the pitch and before the optimal point of the neighbor
                #!This is probably not a good enough check
                if intersect_point[0] < 100 and intersect_point[1] > 0 and intersect_point[1] < 54 and intersect_point[0] > neighbor.optimal_point[0]:
                    print("removed neighbor")
                    #Remove the neighbor, move onto the next neighbor
                    line[1].neighbors.remove(neighbor)
                    break"""

    

    return windows





def get_heuristic(current_node,neighbor,end):
    neighbor.g = (np.linalg.norm(neighbor.optimal_point - current_node.optimal_point)/7)/neighbor.optimal_time
    #print(angle(neighbor.optimal_point, current_node.optimal_point)/(2*math.pi))
    neighbor.g = neighbor.g*angle(neighbor.optimal_point, current_node.optimal_point)
    #neighbor.g = current_node.g + neighbor.optimal_time

    #h is the distance from the neighbor node to the end, only in the x direction
    neighbor.h = np.linalg.norm(neighbor.optimal_point[0] - end[0])
    neighbor.f = neighbor.g/neighbor.h
    return neighbor

def reconstruct_path(current_node, end, start_window,carrier):
    the_path = []
    the_path.append(Window(None,None,end))
    while current_node.parent != None:
        the_path.append(current_node)
        current_node = current_node.parent
    the_path.append(start_window)
    the_path.append(Window(None,None,carrier))
    return the_path

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

            the_path = reconstruct_path(current_node,end,start_window,carrier)
            return the_path
            """the_path = []
            the_path.append(Window(None,None,end))
            while current_node.parent != None:
                the_path.append(current_node)
                current_node = current_node.parent
            the_path.append(start_window)
            the_path.append(Window(None,None,carrier))

            return the_path"""

        for neighbor in current_node.neighbors:
            if neighbor.parent == None:
                neighbor.parent = current_node

            #for all the neighbors for a window, check if they are already in the closed list, if so, do nothing
            if neighbor in closed_list:
                continue

            #if the neighbor is not in the closed list, need to calculate the heuristic
            neighbor = get_heuristic(current_node,neighbor,end)

            #if this neighbor is in the open list already, and it's g value in the open list is less than the g value just calculated, do nothing
            #because the neighbor in the open list is better
            if neighbor in open_list:
                if neighbor.g >= open_list[open_list.index(neighbor)].g:
                    continue
                else:
                    open_list[open_list.index(neighbor)] = neighbor
            #else, add it to the open list
            else:
                open_list.append(neighbor)

    return reconstruct_path(current_node,end,start_window,carrier)

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
