import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import unicodedata
import re
from helpers import get_more_specific_df, create_new_folder, inputpath, outputpath, get_play_description_from_number


'''
The following code can be used to produce an animation for a specific play in a
specific game. This animation can then be saved locally as a html file.

The plays and games can be looped through to produce multiple animations. Below
is an example of how to use the functions in the code. You may need to install
some of the packages above before using the code.
'''


def slugify(path, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    if allow_unicode:
        path = unicodedata.normalize('NFKC', path)
    else:
        path = unicodedata.normalize('NFKD', path).encode('ascii', 'ignore').decode('ascii')
    path = re.sub(r'[^\w\s-]', '', path.lower())
    return re.sub(r'[-\s]+', '-', path).strip('-_')

def animation(csv, play_no, year, path=None, save_html=False, plotting=False):
    fig = px.scatter(csv, x='x', y='y', hover_name='displayName', color='team', animation_frame='frameId', range_x=[0, 120], range_y=[0, 53.3])
    # The plotly only needs the dataframe and some arguments to plot the positions
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 50
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 20
    if plotting == True:
        fig.show()
    if save_html == True:
        if path is not None:
            if len(path) < 200:
                writing_path = path
            else:
                writing_path = path[:200]
                # Only a certain length of string is allowed
            fig.write_html(outputpath+'Tracking_' + year + '\\Play' + str(play) + '\\' + writing_path + '.html')
            # Save the html of the plotly file
    return

def run_anim(year, playIndex):
    year = '2020'
    # Specifiy which year csv file to use
    plays_csv = pd.read_csv(inputpath+'plays.csv')
    # The csv for all the separate plays that have been used throughout all NFL games
    tracking_csv = pd.read_csv(inputpath+'tracking' + year + '.csv')
    # The csv with all the tracking data. All instances of a certain play can be taken from this
    plays = np.unique(tracking_csv['playId'])
    # All the different plays
    play = plays[playIndex]
    csv = get_more_specific_df(tracking_csv, 'playId', play)
    # A csv containing tracking data of all instances of the specific play
    games = np.unique(csv['gameId'])
    # All the games that the specific play has been used in
    game = games[0]
    # Generic 1st game
    new_csv = get_more_specific_df(csv, 'gameId', game)
    # A csv containing tracking data of all instances of the specific play used in a specific game
    save_string = get_play_description_from_number(plays_csv, play, game).iloc[0]
    # Generate the name of the html if you choose to save it
    create_new_folder(outputpath+'Tracking_' + year + '\\Play' + str(play))
    # Initialise a folder for an animation to be added to
    animation(new_csv, play, year, path=slugify(save_string), save_html=True, plotting=True)
    # Run the animation

run_anim('2020', 0)