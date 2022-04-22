from turtle import done
from visualisation_functions import process_play
from helpers import input_folderpath
import os
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import random

output_results = "results.csv"

results = pd.read_csv(output_results)

def processing(play):
    if play in results.values:
        return
    print(play)
    process_play(input_folderpath+"/receiving_plays/"+play,play,output_results)

def create_graph():
    df = pd.read_csv(output_results)
    df = df.dropna()
    df = df.drop(df[df.yardage <= 0].index)
    
    yardage = df['yardage']
    log_yardage = np.log10(yardage)
    median_deviation = df['median_deviation']
    mean_deviation = df['mean_deviation']
    r = np.corrcoef(log_yardage, median_deviation)
    print(r)
    plt.scatter(median_deviation,yardage,s=10)
    m, b = np.polyfit(median_deviation, yardage, 1)
    plt.plot(median_deviation, m*median_deviation+b,c="black")
    plt.title("Plot showing median path deviation vs yards gained for each play")
    plt.xlabel("Median Path Deviation")
    plt.ylabel("Yards gained")
    plt.show()

def process_plays():
    #Change this number depending on how many cores you want to use for this
    pool = Pool()
    pool.map(processing,os.listdir(input_folderpath+"/receiving_plays"))


if __name__ == '__main__':

    #process_plays()
    create_graph()



        