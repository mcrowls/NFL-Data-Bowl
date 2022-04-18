from turtle import done
from visualisation_functions import process_play
from helpers import input_folderpath
import os
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv("results.csv")

def processing(play):
    if play in results.values:
        return
    print(play)
    process_play(input_folderpath+"/receiving_plays/"+play,play)

def create_graph():
    df = pd.read_csv("results.csv")
    df = df.dropna()
    df = df.drop(df[df.yardage < 0].index)
    
    yardage = df['yardage']
    deviation = df['deviation']
    r = np.corrcoef(yardage, deviation)
    print(r)
    plt.scatter(deviation,yardage)
    plt.show()

def process_plays():
    #Change this number depending on how many cores you want to use for this
    pool = Pool(8)
    pool.map(processing,os.listdir(input_folderpath+"/receiving_plays"))


if __name__ == '__main__':

    process_plays()
    create_graph()



        