# NFL-Data-Bowl

Coursework for the Applied Data Science unit analysing, inferring and visualising the NFL Big Data Bowl 2022 Comptetition data.

The NFL Big Data Bowl 2022 data is here [], and the Madden 2021 data is here []. This project aims to analyse Punt Returns, specifically the
Punt Returner's path, the method used to calculate this, and how it can be optimised.

### The repo contains the following notebooks:

* NFL_Data_Cleaning: A notebook running through the data cleaning process for the NFL Big Data Bowl 2022 data
* NFL_Data_Visualise: A notebook containing scripts to save a play as an .mp4 or .html file, alternatively use 'visualisation_functions.py'
* NFL_Data_Results: A notebook showcasing the results in the paper and the comparisons of the techniques used for optimal path calculation

### And the following scripts, with the options commented above

```py
# --all / -a: processes all punt returns (plays)  
# OR 
# --playid / -p: id of the play to process (e.g. play2365-game2020110108)
#
# --inpath / -i: path to the folder containing the play csv's   
# --outpath / -o : path to store the results (/results or /visualisations)
# --logpath / -l : path to store the output - print statements from the logger
# --mem / -m : proportion (% / 100) in range [0, 1] of system memory available during processing (default=0.8=80%)
# --astar_old / -q : boolean whether to use the project's Delaunay window implementation of A* (default, False) 
#                    or to use the standard A* pathfinding algorithm (True)
processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q



# --playid / -p: id of the play to process (e.g. play2365-game2020110108)
# --inpath / -i: path to the folder containing the play csv's   
# --outpath / -o: path to store the results (/results or /visualisations)
# --visfunc / -v: function to use to create the play visualisation; new/old/funcanim
# --astar_old / -q: function to use to create the play visualisation; new/old/funcanim
visualisation_functions.py -p <play_id> -i <input_path> -o <output_path> -v <"new"/"old"/"funcanim"> -q
```

Global variables, input/output paths and defaults for command-line options can be changed in helpers.py:

```py
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
```

### There are also the following commands which setup the environment (dependencies, notebook options):

```py
pip install -U tk
pip install -U jupyter_console
pip install -U ipywidgets
pip install -U ffmpeg-python
jupyter nbextension enable --py widgetsnbextension
```