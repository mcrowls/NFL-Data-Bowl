import os
import numpy as np

"""# Global Variables / Paths"""

"""## Edit These"""
input_folderpath = "data"
output_folderpath = "data"
play_filename = 'play2365-game2020110108.csv'
vis_output_folderpath = "visualisations"
results_output_folderpath = "results"
drive_folderpath = "Colab Notebooks"
useDrive = False
avg_player_speed = 7
num_threads = 16

heuristic_func = lambda c, d: np.linalg.norm(c[0] - d[0])



"""## DO NOT EDIT These"""
drivepath = 'drive/MyDrive/'+drive_folderpath+"/"
inputpath = drivepath+input_folderpath+"/" if useDrive else input_folderpath+"/"
outputpath = drivepath+output_folderpath+"/" if useDrive else output_folderpath+"/"
visoutputpath = drivepath+vis_output_folderpath+"/" if useDrive else vis_output_folderpath+"/"
resultsoutputpath = drivepath+results_output_folderpath+"/" if useDrive else results_output_folderpath+"/"
playpath = inputpath+"receiving_plays/"+play_filename
#drivepath = 'drive\\MyDrive\\'+drive_folderpath+"\\"
#inputpath = drivepath+input_folderpath+"\\" if useDrive else input_folderpath+"\\"
#outputpath = drivepath+output_folderpath+"\\" if useDrive else output_folderpath+"\\"

"""# Inside visualisation_functions, use these"""
# visualise_play_delaunay(playpath)
# visualise_play(playpath)

def get_more_specific_df(df, column, value):
    df = df[df[column] == value]
    # Returns a df where all values of a certain column are a certain value
    return df

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # Only create the folder if it is not already there
    return

def get_play_description_from_number(csv, play_no, game_id):
    play = get_more_specific_df(csv, 'playId', play_no)
    specific_play = get_more_specific_df(play, 'gameId', game_id)
    description = specific_play['playDescription']
    # Extracting the play description if we are to save the animation locally
    return description