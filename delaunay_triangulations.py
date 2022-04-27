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
from helpers import avg_player_speed
import statistics
from players import Player
from frechetdist import frdist
import copy

class Window:
    def __init__(self, points, optimal_time,optimal_point,triangle = [],start=[],end=[],direction=None):
        self.points = points
        self.optimal_time = optimal_time
        self.optimal_point = optimal_point
        self.triangle = triangle
        self.neighbors = [] #a list of windows
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.start = start
        self.end = end
        self.direction = direction

def finding_five_yard(array):
    if len(array) == 1:
        return array
    starting_point = array[0]
    relevant_points = [array[0]]
    i = 1
    truth = False
    while truth == False:
        if abs(array[i][0] - starting_point[0]) < 5:
            relevant_points.append(array[i])
        else:
            relevant_points.append(array[i])
            truth = True
        i += 1
        #If the path is less than 5 yards in total, need to stop 
        if i == len(array):
            truth = True
    return relevant_points


def find_five_point(array):
    point1 = array[-2]
    point2 = array[-1]
    gradient = (point2[1] - point1[1])/(point2[0] - point1[0])
    intercept = point2[1] - gradient*point2[0]
    return [array[0][0] - 5, gradient*(array[0][0] - 5) + intercept]


def path_interpolate(array, n):
    distances_array = []
    for i in range(np.shape(array)[0] - 1):
        distances_array.append(np.sqrt((array[i][0] - array[i+1][0])**2 + (array[i][1] - array[i+1][1])**2))
    ratios = []
    for distance in distances_array:
        ratios.append(round(distance/np.sum(distances_array)*n))
    points = []
    for i in range(np.size(ratios)):
        segment = np.linspace(array[i], array[i+1], ratios[i])
        for element in segment:
            points.append(element)
    return points

def frechet_distance(actual_path, predicted_path):
    # cut off both of these after 5 yards moved in the x direction
    new_predicted_path = []
    for j in reversed(predicted_path):
        new_predicted_path.append(j)
    actual_path = finding_five_yard(actual_path)
    predicted_path = finding_five_yard(new_predicted_path)
    #actual_path[-1] = find_five_point(actual_path)
    predicted_path[-1] = find_five_point(predicted_path)
    actual_path = path_interpolate(actual_path, 50)
    #plt.scatter(np.array(actual_path)[:,0],np.array(actual_path)[:,1])
    predicted_path = path_interpolate(predicted_path, 50)
    #plt.scatter(np.array(predicted_path)[:,0],np.array(predicted_path)[:,1])

    #Sometimes the actual path isn't at least 5 yards long, so this just shortens the predicted path
    if (np.array(actual_path).shape[0] < np.array(predicted_path).shape[0]):
        actual_shape = np.array(actual_path).shape[0]
        predicted_shape = np.array(predicted_path).shape[0]
        difference = predicted_shape - actual_shape
        predicted_path = predicted_path[:-difference]
    elif (np.array(actual_path).shape[0] > np.array(predicted_path).shape[0]):
        actual_shape = np.array(actual_path).shape[0]
        predicted_shape = np.array(predicted_path).shape[0]
        difference = actual_shape - predicted_shape
        actual_path = actual_path[:-difference]
    return frdist(actual_path, predicted_path)

def distance(loc1, loc2):
    return np.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)

def angle(point1, point2):
    if (point2[0] - point1[0]) == 0:
        return 0
    return abs(np.arctan((point2[1] - point1[1])/(point2[0] - point1[0])))

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

def get_lines_from_sidelines(top,left,right,returner_pos):
    points = []
    windows = []
    for t in top:
        line = np.linspace(t,[returner_pos[0],t[1]],22,endpoint=True)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=t,end=line[-1],direction="t"))

    for l in left:
        line = np.linspace(l,[l[0], 53.3],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=l,end=line[-1],direction="l"))

    for r in right:
        line = np.linspace(r,[r[0], 0 ],22,endpoint=False)[1:]
        points.append(line)
        windows.append(Window(line,0,[0,0],start=r,end=line[-1],direction="r"))

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
def get_arrival_times(windows,side_windows,defenders, blockers, frame,returner_pos):
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
    line = np.linspace([returner_pos[0],0],[returner_pos[0],53],22,endpoint=False)[1:]
    updated_windows.append(Window(line,0,np.array(returner_pos),start=[returner_pos[0],0],end=[returner_pos[0],53],direction="start"))
    times.extend([0 for i in range(21)])
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
                if other_window.direction == "start":
                    other_window.neighbors.append(window)
                for triangle in window.triangle:
                    if triangle in other_window.triangle and other_window.optimal_point[0]<= window.optimal_point[0]:
                        window.neighbors.append(other_window)
                        
                #giving delaunay windows sideline neighbors
                #if the delaunay window has the same start point as the sideline window, they should be neighbors
                #! But this causes problems because it causes some window neighbors to have other windows in between them
                #! Which is why the check_window_neighbors was created
                if len(other_window.triangle) == 0 and len(window.triangle) ==  1:
                    if other_window.optimal_point[0]<= window.optimal_point[0]:
                        window.neighbors.append(other_window)
            #this is to give sideline windows any neighbors
            else:
                #matching sidelines to delaunay windows
                #if the delaunay window has the same start point as the sideline window, they should be neighbors
                #! But this causes problems because it causes some window neighbors to have other windows in between them
                #! Which is why the check_window_neighbors was created
                if len(other_window.triangle) == 1 and other_window.optimal_point[0]<= window.optimal_point[0]:
                    window.neighbors.append(other_window)

    #this gives sideline windows sideline neighbors
    tops = []
    lefts = []
    rights = []
    start = None
    for window in windows:
        if window.direction == "start":
            start = window
    #create lists of all the sideline window types
    for window in windows:
        if window.direction == "t":
            tops.append(window)
        elif window.direction == "l":
            lefts.append(window)
        elif window.direction == "r":
            rights.append(window)

    start.neighbors.extend(tops)
    start.neighbors.extend(lefts)
    start.neighbors.extend(rights)
    #sort them based on the optimal point
    tops.sort(key=lambda x: x.optimal_point[1], reverse=True)
    lefts.sort(key=lambda x: x.optimal_point[0], reverse=True)
    rights.sort(key=lambda x: x.optimal_point[0], reverse=True)

    #give each sideline window it's adjacent sideline windows as neighbors
    for top in tops[1:len(tops)-1]:
        prev = tops[tops.index(top)-1]
        next = tops[tops.index(top)+1]
        if prev.optimal_point[0] > top.optimal_point[0]:
            top.neighbors.append(prev)
        if next.optimal_point[0] > top.optimal_point[0]:
            top.neighbors.append(next)

    if len(tops) > 1 and tops[0].optimal_point[0] < tops[1].optimal_point[0]:
        tops[0].neighbors.append(tops[1])

    if len(tops) > 1 and tops[-1].optimal_point[0] < tops[-2].optimal_point[0]:   
        tops[-1].neighbors.append(tops[-2])

    for left in lefts[1:len(lefts)-1]:
        next = lefts[lefts.index(left)+1]
        left.neighbors.append(next)

    if len(lefts) > 1:
        lefts[0].neighbors.append(lefts[1])

    for right in rights[1:len(rights)-1]:
        next = rights[rights.index(right)+1]
        right.neighbors.append(next)

    if len(rights) > 1:
        rights[0].neighbors.append(rights[1])

    #the first top sideline window should be connected to the nearest left sideline window
    if len(tops) > 1 and len(lefts) > 1:
        tops[0].neighbors.append(lefts[0])

    #the last top sideline window should be connected to the nearest right sideline window
    if len(tops) > 1 and len(rights) > 1:
        tops[-1].neighbors.append(rights[0])

    #Now check for bad neighbors
    windows = check_window_neighbors(windows)
    return windows

def check_intercept(window,other_window,neighbor,slope_w,intercept_w):
    x1 = window.optimal_point[0]
    y1 = window.optimal_point[1]
    x2 = neighbor.optimal_point[0]
    y2 = neighbor.optimal_point[1]
    #the line between the optimal point of a window and it's neighbor's optimal point
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2

    #Finding the point of intersection between the neighbor line and the other window
    if math.isinf(slope_w):
        intercept_x = other_window.start[0]
    elif math.isinf(slope):
        intercept_x = window.optimal_point[0]
    else:
        intercept_x = (intercept - intercept_w) / (slope_w - slope)
        if math.isnan(intercept_x):
            intercept_x = float('inf')

    if math.isinf(slope_w):
        intercept_y = slope * intercept_x + intercept
    elif math.isinf(slope):
        intercept_y = slope_w * intercept_x + intercept_w
    else:
        intercept_y = slope * intercept_x + intercept
    
    if math.isnan(intercept_y):
        intercept_y = float('inf')

    #Notebook code rounded everything, not sure if it's necessary
    intercept_x = round(intercept_x,2)
    intercept_y = round(intercept_y,2)
    x1 = round(window.optimal_point[0],2)
    y1 = round(window.optimal_point[1],2)
    x2 = round(neighbor.optimal_point[0],2)
    y2 = round(neighbor.optimal_point[1],2)
    wx1 = round(other_window.start[0],2)
    wy1 = round(other_window.start[1],2)
    wx2 = round(other_window.end[0],2)
    wy2 = round(other_window.end[1],2)

    is_intersect = ((intercept_x <= x1 and intercept_x >= x2) or (intercept_x >= x1 and intercept_x <= x2)) and \
    ((intercept_y <= y1 and intercept_y >= y2) or (intercept_y >= y1 and intercept_y <= y2)) and \
    ((intercept_x <= wx1 and intercept_x >= wx2) or (intercept_x >= wx1 and intercept_x <= wx2)) and \
    ((intercept_y <= wy1 and intercept_y >= wy2) or (intercept_y >= wy1 and intercept_y <= wy2))

    return is_intersect

def get_window_equation(window):
    #Finding the equation of the other window to check for intersection
    if (window.start[0] - window.end[0] == 0):
        slope_w = float('inf')
    else:
        slope_w = (window.start[1] - window.end[1]) / (window.start[0] - window.end[0])
    if not math.isinf(slope_w):   
        intercept_w = window.start[1] - slope_w * window.start[0]
    else:
        intercept_w = float('inf')
    
    return slope_w,intercept_w


def check_window_neighbors(windows):
    #For each window's neighbors, check if the line between a window and it's neighbour intersect any other window,
    #If so, remove that neighbour
    for window in windows:  
        for other_window in windows:
            if other_window == window or other_window.direction == "start":
                continue
            
            if window.direction == "start" and other_window.direction == "start":
                continue

            slope_w, intercept_w = get_window_equation(other_window)

            sus_neighbors = []
            for neighbor in window.neighbors:

                #Don't check for intersections between delaunay windows and delaunay neighbors as those are all correct
                if len(window.triangle) > 0 and len(neighbor.triangle) > 0:
                    continue
               
                if neighbor == window or neighbor == other_window: 
                    continue
               
                #Don't check intersection between sideline window and sideline neighbor as those are all correct
                if len(window.triangle) == 0 and len(neighbor.triangle) == 0 and not window.direction == "start":
                    continue
                
                is_intersect = check_intercept(window,other_window,neighbor,slope_w,intercept_w)

                if is_intersect:
                    sus_neighbors.append(neighbor)

            for sus in sus_neighbors:
                window.neighbors.remove(sus)
                                 
    return windows


def point_in_rect(point,rect):
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
    if len(points_after_returner) == 0:
        return [], [], []


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
    if (len(ind_top) == 1):
        top_window = [points_after_returner[ind_top[0]]]
    else:
        top_window = [points_after_returner[ind_top[0]], points_after_returner[ind_top[1]]]

    for i in range(0, len(points_x)):
        if points_x[i] > x_lim:
            top_window.append(points_after_returner[i])
        if points_y[i] > y_lim:
            left_window.append(points_after_returner[i])
        if points_y[i] < y_lim:
            right_window.append(points_after_returner[i])

    return np.unique(top_window,axis=0), np.unique(right_window,axis=0), np.unique(left_window,axis=0)

# DEFAULT: np.linalg.norm(neighbor.optimal_point[0] - end[0])
# OLD: np.linalg.norm(neighbor.optimal_point - end)/return_speed
def get_heuristic(current_node,neighbor,end,return_speed,heuristic=lambda c, d: np.linalg.norm(c[0] - d[0]), old_astar=False):
    if old_astar:
        neighbor.g = (np.linalg.norm(neighbor.optimal_point - current_node.optimal_point)/return_speed)
        neighbor.g = current_node.g + neighbor.g
        #h is the distance from the neighbor node to the end, only in the x direction
        h_f = lambda c, d: np.linalg.norm(c - d)
        neighbor.h = h_f(neighbor.optimal_point, end)
        neighbor.f = neighbor.g + neighbor.h
    else:
        neighbor.g = (np.linalg.norm(neighbor.optimal_point - current_node.optimal_point)/return_speed)/neighbor.optimal_time
        #print(angle(neighbor.optimal_point, current_node.optimal_point)/(2*math.pi))
        neighbor.g = neighbor.g*angle(neighbor.optimal_point, current_node.optimal_point)
        #neighbor.g = current_node.g + neighbor.g
        
        #h is the distance from the neighbor node to the end, only in the x direction
        neighbor.h = heuristic(neighbor.optimal_point, end)#np.linalg.norm(neighbor.optimal_point[0] - end[0])
        neighbor.f = neighbor.g/neighbor.h
    return neighbor

def reconstruct_path(current_node, end, start_window,carrier):
    the_path = []
    the_path.append(Window(None,None,[10,current_node.optimal_point[1]]))
    while current_node.parent != None:
        the_path.append(current_node)
        current_node = current_node.parent
        #Sometimes this causes an infinite loop because two nodes have each other as parents
        #so this just kills the loop if that happens. Not ideal behaviour but at least it won't hang
        if current_node in the_path:
            the_path.append(start_window)
            the_path.append(Window(None,None,carrier))
            return the_path
    the_path.append(start_window)
    the_path.append(Window(None,None,carrier))
    return the_path

def get_optimal_path(windows,carrier,end,return_speed, heuristic=lambda c, d: np.linalg.norm(c[0] - d[0]), old_astar=False):
    start_window = None
    #assuming end is an x y point, need to find the window closest to the end
    min_dist_end = float('inf')
    end_window = None
    for window in windows:
        if window.direction == "start":
            start_window = window
        #the end window is the closest window to the end point
        if np.linalg.norm(end[0]-window.optimal_point[0]) < min_dist_end:
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

        for neighbor in current_node.neighbors:
            if neighbor.parent == None:
                neighbor.parent = current_node

            #for all the neighbors for a window, check if they are already in the closed list, if so, do nothing
            if neighbor in closed_list:
                continue

            #if the neighbor is not in the closed list, need to calculate the heuristic
            neighbor = get_heuristic(current_node,neighbor,end,return_speed, heuristic=heuristic, old_astar=old_astar)

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
            attackers.append(Player(player, player_csv['x'], player_csv['y'], team))
        else:
            defenders.append(Player(player, player_csv['x'], player_csv['y'], team))
    return attackers, defenders