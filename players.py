import numpy as np
import pandas as pd
from helpers import inputpath, avg_player_speed


class Player:
    def __init__(self, name, xs, ys, team, speed=None):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.team = team
        try:
            csv = pd.read_csv(inputpath+'player_speeds.csv')
            s = get_fuzzy_match_speed(self.name, csv)
            if s is not None:
                self.speed = s
            else:
                self.speed = avg_player_speed
        except:
            self.speed = avg_player_speed
        if speed is not None:
            self.speed = speed

    def getxyloc(self, i):
        x = self.xs.iloc[i]
        y = self.ys.iloc[i]
        return [x, y]
    
    def get_speed(self):
        csv = pd.read_csv(inputpath+'player_speeds.csv')
        try:
            csv = pd.read_csv(inputpath+'player_speeds.csv')
            s = get_fuzzy_match_speed(self.name, csv)
            if s is not None:
                speed = s
            else:
                speed = avg_player_speed
        except:
            speed = avg_player_speed
        return speed
    
def get_player_speed(playername):
    csv = pd.read_csv(inputpath+'player_speeds.csv')
    try:
        csv = pd.read_csv(inputpath+'player_speeds.csv')
        s = get_fuzzy_match_speed(playername, csv)
        if s is not None:
            return s
        else:
            return avg_player_speed
    except:
        return avg_player_speed
    
def get_fuzzy_match_speed(playername, playercsv, T=0.8):
    from fuzzywuzzy import fuzz
    # 0 - 100
    potential_names = []
    ratios = []
    for name in playercsv['Name'].tolist():
        fr = fuzz.ratio(name,playername)
        if fr >= T*100:
            potential_names.append(name)
            ratios.append(fr)
    if len(potential_names) == 0:
        return None
    else:
        zipped = zip(potential_names, ratios)
        zipped = list(zipped)
        res = sorted(zipped, key = lambda x: x[1]).reverse()
        return res[0][0]