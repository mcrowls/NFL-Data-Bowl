import os

"""# Global Variables / Paths"""

"""## Edit These"""
input_folderpath = "data"
output_folderpath = "data"
play_folderpath = 'receiving_plays/play116-game2021010301.csv'
drive_folderpath = "Colab Notebooks"
useDrive = False


"""## DO NOT EDIT These"""
drivepath = 'drive/MyDrive/'+drive_folderpath+"/"
inputpath = drivepath+input_folderpath+"/" if useDrive else input_folderpath+"/"
outputpath = drivepath+output_folderpath+"/" if useDrive else output_folderpath+"/"
playpath = inputpath+play_folderpath
#drivepath = 'drive\\MyDrive\\'+drive_folderpath+"\\"
#inputpath = drivepath+input_folderpath+"\\" if useDrive else input_folderpath+"\\"
#outputpath = drivepath+output_folderpath+"\\" if useDrive else output_folderpath+"\\"

"""# Inside visualisation_functions, use these"""
# visualise_delaunay_play(playpath)
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
