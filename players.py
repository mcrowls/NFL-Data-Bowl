import numpy as np
import pandas as pd
from helpers import inputpath, avg_player_speed


class Player:
    def __init__(self, name, xs, ys, team, speed):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.team = team
        try:
            csv = pd.read_csv(inputpath+'player_speeds.csv')
            player = csv[csv['Name'] == self.name]
            speed = player['Speed'][0]
            self.speed = speed
        except:
            self.speed = avg_player_speed
        

    def getxyloc(self, i):
        x = self.xs.iloc[i]
        y = self.ys.iloc[i]
        return [x, y]
    
    def get_speed(self):
        csv = pd.read_csv(inputpath+'player_speeds.csv')
        try:
            player = csv[csv['Name'] == self.name]
            speed = player['Speed'][0]
            return speed
        except:
            speed = avg_player_speed
            return speed
    
def get_player_speed(playername):
    csv = pd.read_csv(inputpath+'player_speeds.csv')
    #TODO FUZZY MATCHING
    try:
        player = csv[csv['Name'] == playername]
        speed = player['Speed'][0]
        return speed
    except:
        return avg_player_speed