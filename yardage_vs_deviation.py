from turtle import done
from visualisation_functions import process_play
from helpers import input_folderpath
import os
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import random

output_results = "results-after-distance-fix.csv"

results = pd.read_csv(output_results)

def processing(play):
    if play in results.values:
        return
    print(play)
    process_play(input_folderpath+"/receiving_plays/"+play,play,output_results)

def create_graph():
    df = pd.read_csv(output_results)
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
    pool = Pool()
    pool.map(processing,random.sample(os.listdir(input_folderpath+"/receiving_plays"),50))


if __name__ == '__main__':

    process_plays()
    #create_graph()



        