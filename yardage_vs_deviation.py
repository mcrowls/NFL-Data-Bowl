from turtle import done
from visualisation_functions import process_play
from helpers import input_folderpath
import os
import pandas as pd

results = pd.read_csv("results.csv")
done_plays = results['play']
print(done_plays)

for play in os.listdir(input_folderpath+"/receiving_plays"):
    #skip already done plays
    if play in results.values:
        continue
    print(play)
    process_play(input_folderpath+"/receiving_plays/"+play,play)