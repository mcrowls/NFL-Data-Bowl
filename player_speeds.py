import numpy as np
import pandas as pd


def get_speed(csv, player_name):
    player = csv[csv['Name'] == player_name]
    speed = player['Speed'][0]
    return speed


csv = pd.read_csv('csvs/PlayerSpeeds.csv')
