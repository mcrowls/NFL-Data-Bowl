import numpy as np
import pandas as pd
from helpers import inputpath


class Player:
    def __init__(self, name, xs, ys, team, speed):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.team = team
        self.speed = speed

    def getxyloc(self, i):
        x = self.xs.iloc[i]
        y = self.ys.iloc[i]
        return [x, y]

    def get_speed(self):
        csv = pd.read_csv(inputpath+'Speed_Data.csv')
        player = csv[csv['Name'] == self.name]
        speed = player['Speed'][0]
        return speed

def get_player_speeds(foldername):
    csv = pd.read_csv(inputpath+'Speed_Data.csv')
    return csv
